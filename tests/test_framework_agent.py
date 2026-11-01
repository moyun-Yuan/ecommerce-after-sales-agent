"""框架 Agent 的权限、预算、消息回传与模型协议测试，不请求外部服务。"""
from __future__ import annotations

import gc
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from ecommerce_after_sales_agent import config, db, graph, rag, seed, tools
from ecommerce_after_sales_agent.llm import create_chat_model
from ecommerce_after_sales_agent.solver_agent import LIMIT_REPLY, invoke_solver_agent
from ecommerce_after_sales_agent.usage import TokenUsageTracker


class RecordingModel(FakeMessagesListChatModel):
    """使用真实聊天模型基类执行框架链路，仅将模型生成替换为预设消息。"""

    bound_names: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, active_tools, **kwargs):
        self.bound_names.append([item.name for item in active_tools])
        return self


def tool_request(name, args, call_id="call-one"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class FrameworkAgentTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="framework-agent-", dir=cache)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(config, "DB_PATH", Path(self.temporary.name) / "business.sqlite"))
        self.stack.enter_context(patch.dict(os.environ, {"SOLVER_THINKING": "false"}))
        with redirect_stdout(io.StringIO()):
            seed.seed_sqlite()

    def tearDown(self):
        self.stack.close()
        gc.collect()
        self.temporary.cleanup()

    def state(self, route="use_tools"):
        return {"thread_id": "framework-thread", "customer_id": "CUST-1001", "route": route,
                "messages": [HumanMessage(content="请帮我处理订单 ORD-5001")], "retrieved": []}

    def solve(self, responses, route="use_tools", **state_changes):
        model = RecordingModel(responses=responses)
        with patch.object(graph, "_llm", return_value=model):
            result = graph.solver_node({**self.state(route), **state_changes})
        return result, model

    def test_direct_route_only_binds_policy_and_rejects_forged_tools(self):
        denied = [tool_request("get_order", {"order_id": "ORD-5001"}, "order"),
                  tool_request("create_refund_draft", {"order_id": "ORD-5001", "amount": 50}, "draft"),
                  tool_request("issue_refund", {"order_id": "ORD-5001", "amount": 50}, "refund"),
                  tool_request("send_reply", {"text": "已退款"}, "send")]
        result, model = self.solve([AIMessage(content="", tool_calls=denied), AIMessage(content="需要具体订单操作时请补充申请。")],
                                   route="answer_directly")
        self.assertTrue(all(names == ["check_refund_policy"] for names in model.bound_names))
        self.assertTrue(all(item["result"]["status"] == "rejected" for item in result["tool_calls"]))
        self.assertIsNone(result["proposed_action"])
        with closing(db.get_conn()) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM actions_log").fetchone()[0], 0)

    def test_policy_tool_result_maps_back_to_outer_state(self):
        policy = {"doc_id": "refund-policy", "title": "退款政策", "text": "执行需人工批准。"}
        with patch.object(rag, "retrieve", return_value=[policy]):
            result, _ = self.solve([
                AIMessage(content="", tool_calls=[tool_request("check_refund_policy", {"query": "部分退款"})]),
                AIMessage(content="需人工批准。[refund-policy]"),
            ], route="answer_directly")
        self.assertEqual(result["retrieved"], [policy])
        self.assertEqual(result["citations"], ["refund-policy"])
        self.assertEqual(result["tool_calls"][0]["result"]["results"], [policy])

    def test_tool_validation_error_returns_message_without_creating_draft(self):
        result, _ = self.solve([
            AIMessage(content="", tool_calls=[tool_request("create_refund_draft", {"order_id": "ORD-5001", "amount": True})]),
            AIMessage(content="请提供有效退款金额。"),
        ])
        self.assertEqual(result["tool_calls"][0]["result"]["status"], "rejected")
        self.assertIsNone(result["proposed_action"])

    def test_model_call_limit_ends_repeated_tools_with_chinese_reply(self):
        model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
            tool_request("get_order", {"order_id": "ORD-5001"}, f"order-{index}")
        ]) for index in range(8)])
        messages = invoke_solver_agent(model, [tools.get_order], [HumanMessage(content="查询订单")],
                                       thread_id="budget", customer_id="CUST-1001", max_model_calls=3)
        self.assertEqual(model.i, 3)
        self.assertEqual(sum(isinstance(item, ToolMessage) for item in messages), 3)
        self.assertEqual(messages[-1].content, LIMIT_REPLY)

    def test_oversized_tool_batch_stops_before_any_tool_executes(self):
        model = RecordingModel(responses=[AIMessage(content="", tool_calls=[
            tool_request("get_order", {"order_id": "ORD-5001"}, f"order-{index}") for index in range(3)
        ])])
        messages = invoke_solver_agent(model, [tools.get_order], [HumanMessage(content="查询订单")],
                                       thread_id="batch-budget", customer_id="CUST-1001", max_tool_calls=2)
        self.assertEqual(messages[-1].content, LIMIT_REPLY)
        self.assertEqual(sum(isinstance(item, ToolMessage) and item.status == "error" for item in messages), 3)
        with closing(db.get_conn()) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM actions_log").fetchone()[0], 0)

    def test_batch_executes_in_order_and_only_creates_draft(self):
        result, _ = self.solve([
            AIMessage(content="", tool_calls=[
                tool_request("get_order", {"order_id": "ORD-5001"}, "order"),
                tool_request("create_refund_draft", {"order_id": "ORD-5001", "amount": 50}, "draft"),
            ]), AIMessage(content="50元草案等待审批。"),
        ])
        self.assertEqual(result["proposed_action"]["amount"], 50)
        self.assertTrue(result["proposed_action"]["request_id"])
        with closing(db.get_conn()) as conn:
            self.assertEqual([row[0] for row in conn.execute("SELECT action FROM actions_log ORDER BY id")],
                             ["get_order", "create_refund_draft"])
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM approvals").fetchone()[0], 0)

    def test_tool_worker_restores_customer_and_audit_context(self):
        tools.set_run_context("wrong-thread", "CUST-1002")
        model = RecordingModel(responses=[
            AIMessage(content="", tool_calls=[tool_request("get_order", {"order_id": "ORD-5001"})]),
            AIMessage(content="查询完成。"),
        ])
        messages = invoke_solver_agent(model, [tools.get_order], [HumanMessage(content="查询订单")],
                                       thread_id="correct-thread", customer_id="CUST-1001")
        output = next(item for item in messages if isinstance(item, ToolMessage))
        self.assertEqual(json.loads(output.content)["customer_id"], "CUST-1001")
        with closing(db.get_conn()) as conn:
            row = conn.execute("SELECT thread_id, customer_id FROM actions_log").fetchone()
            self.assertEqual(tuple(row), ("correct-thread", "CUST-1001"))

    def test_review_retry_removes_draft_tool_and_blocks_forged_call(self):
        result, model = self.solve([
            AIMessage(content="", tool_calls=[tool_request("create_refund_draft", {"order_id": "ORD-5001", "amount": 50})]),
            AIMessage(content="此前退款已执行。"),
        ], approval_decision="approved", action_result={"order_id": "ORD-5001", "amount": 100})
        self.assertTrue(all("create_refund_draft" not in names for names in model.bound_names))
        self.assertEqual(result["tool_calls"][0]["result"]["status"], "rejected")
        self.assertIsNone(result["proposed_action"])

    @staticmethod
    def http_response(message):
        return httpx.Response(200, json={
            "id": "agent-local-test", "object": "chat.completion", "created": 0, "model": "local-test",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17,
                      "completion_tokens_details": {"reasoning_tokens": 3}},
        })

    def assert_http_round_trip(self, thinking):
        requests = []
        tracker = TokenUsageTracker()

        def transport(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertNotIn("response_format", payload)
            self.assertEqual(payload["thinking"]["type"], "enabled" if thinking else "disabled")
            if len(requests) == 1:
                message = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "order-one", "type": "function", "function": {
                        "name": "get_order", "arguments": json.dumps({"order_id": "ORD-5001"}),
                    },
                }]}
                if thinking:
                    message["reasoning_content"] = "本地推理占位。"
                return self.http_response(message)
            assistant = next(item for item in payload["messages"] if item["role"] == "assistant")
            if thinking:
                self.assertEqual(assistant["reasoning_content"], "本地推理占位。")
            tool_message = next(item for item in payload["messages"] if item["role"] == "tool")
            self.assertEqual(tool_message["tool_call_id"], "order-one")
            self.assertEqual(json.loads(tool_message["content"])["id"], "ORD-5001")
            message = {"role": "assistant", "content": "查询完成。"}
            if thinking:
                message["reasoning_content"] = "本地推理结束。"
            return self.http_response(message)

        with patch.object(config, "DEEPSEEK_API_KEY", "local-test-placeholder"), \
             patch.object(config, "DEEPSEEK_BASE_URL", "https://local-test.invalid/v1"), \
             patch.dict(os.environ, {"SOLVER_THINKING": str(thinking).lower()}), \
             httpx.Client(transport=httpx.MockTransport(transport)) as client:
            model = create_chat_model(model="local-test", role="solver", http_client=client)
            with patch.object(graph, "_llm", return_value=model), \
                 patch.object(graph, "triage_node", return_value={"route": "use_tools"}), \
                 patch.object(graph, "reviewer_node", return_value={"review": {"approved": True}}):
                # 外层图回调应自动进入内层 Agent，不能丢失节点或重复统计。
                app = graph.build_graph(InMemorySaver())
                result = graph.run_turn("请查询订单 ORD-5001", "CUST-1001", "http-agent",
                                        app=app, callbacks=[tracker])["state"]
        self.assertEqual(result["proposed_response"], "查询完成。")
        self.assertEqual(len(requests), 2)
        self.assertEqual(tracker.summary()["totals"]["calls"], 2)
        self.assertEqual(tracker.summary()["totals"]["total_tokens"], 34)
        self.assertIn("solver:" + ("enabled" if thinking else "disabled"), tracker.summary()["by_stage"])
        self.assertNotIn("本地推理占位", json.dumps(tracker.summary(), ensure_ascii=False))

    def test_framework_agent_preserves_disabled_mode_protocol_and_usage(self):
        self.assert_http_round_trip(False)

    def test_framework_agent_preserves_thinking_tool_history_and_usage(self):
        self.assert_http_round_trip(True)


if __name__ == "__main__":
    unittest.main()
