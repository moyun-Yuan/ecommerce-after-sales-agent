"""结构化输出的字段校验、失败分支及模型调用隔离测试，全部使用本地替身。"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

import eval as evaluation
from ecommerce_after_sales_agent import config, graph
from ecommerce_after_sales_agent.llm import create_chat_model, invoke_structured
from ecommerce_after_sales_agent.schemas import CorrectnessDecision, ReviewDecision, TriageDecision


class JsonModel:
    """输出指定 JSON，再交给真实 Pydantic 模型校验。"""

    def __init__(self, payload):
        self.payload = payload

    def with_structured_output(self, schema, *, method):
        return RunnableLambda(lambda messages: schema.model_validate_json(self.payload))


class StructuredOutputTests(unittest.TestCase):
    def test_triage_requires_fields_and_valid_route(self):
        """缺少分类、非法枚举和多余字段都不能取得有效分类对象。"""
        invalid = [
            {"reason": "缺少分类"},
            {"route": "refund_now", "reason": "非法分类"},
            {"route": "use_tools"},
            {"route": "use_tools", "reason": "正常", "extra": True},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                TriageDecision.model_validate(payload)

    def test_string_boolean_is_not_accepted(self):
        """防止 bool('false') 为 True 的错误放行。"""
        payload = {"grounded": True, "citations_ok": True, "pii_found": False,
                   "approved": "false", "reason": "不应放行"}
        with self.assertRaises(ValidationError):
            ReviewDecision.model_validate(payload)

    def test_invalid_triage_escalates_instead_of_granting_tools(self):
        for payload in ('{}', '{"route":"refund_now","reason":"错误"}', '不是 JSON'):
            with self.subTest(payload=payload), patch.object(graph, "_llm", return_value=JsonModel(payload)):
                state = graph.triage_node({})
            self.assertEqual(state["route"], "escalate_to_human")
            self.assertEqual(graph._route_after_triage(state), "respond")

    def test_valid_triage_preserves_classification(self):
        model = JsonModel('{"route":"use_tools","reason":"查询具体订单"}')
        with patch.object(graph, "_llm", return_value=model):
            state = graph.triage_node({"messages": [HumanMessage(content="查订单 ORD-5001")]})
        self.assertEqual(state["route"], "use_tools")
        self.assertEqual(state["triage_reason"], "查询具体订单")

    def test_invalid_review_retries_then_routes_to_final_response(self):
        state = {"retrieved": [{"doc_id": "refund-policy", "text": "退款需审批。"}],
                 "proposed_response": "已经退款。[refund-policy]", "retries": 0}
        with patch.object(graph, "_llm", return_value=JsonModel('{}')):
            first = graph.reviewer_node(state)
            second = graph.reviewer_node({**state, **first})
        self.assertFalse(first["review"]["approved"])
        self.assertEqual(graph._route_after_review(first), "solver")
        self.assertEqual(graph._route_after_review(second), "respond")

    def test_contradictory_review_is_not_approved(self):
        payload = json.dumps({"approved": True, "grounded": False, "citations_ok": True,
                              "pii_found": False, "reason": "缺少依据"})
        with patch.object(graph, "_llm", return_value=JsonModel(payload)):
            result = graph.reviewer_node({"retrieved": [{"doc_id": "p", "text": "政策"}],
                                          "proposed_response": "答复[p]"})
        self.assertFalse(result["review"]["approved"])

    def test_invalid_evaluation_is_not_passed(self):
        model = JsonModel('{"pass":"false","reason":"字符串布尔值"}')
        with patch.object(evaluation, "create_chat_model", return_value=model):
            self.assertFalse(evaluation.judge_correctness("问题", "标准", "答复"))
        self.assertEqual(CorrectnessDecision.model_validate({"pass": False, "reason": "不符合"})
                         .model_dump(by_alias=True)["pass"], False)

    def test_network_errors_are_not_silenced_as_format_failures(self):
        class BrokenModel:
            def with_structured_output(self, schema, *, method):
                def fail(messages):
                    raise ConnectionError("本地模拟网络错误")
                return RunnableLambda(fail)
        with self.assertRaises(ConnectionError):
            invoke_structured(BrokenModel(), TriageDecision, [])

    def test_real_langchain_payload_keeps_modes_separate(self):
        """经过真实 LangChain 包装和 HTTP 序列化，只把传输替换成本地 MockTransport。"""
        requests = []

        def transport(request):
            payload = json.loads(request.content)
            requests.append(payload)
            content = ('{"route":"answer_directly","reason":"普通问候"}'
                       if payload.get("response_format") else "你好，请问需要什么帮助？")
            return httpx.Response(200, json={
                "id": "local-test", "object": "chat.completion", "created": 0,
                "model": "local-test", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        with httpx.Client(transport=httpx.MockTransport(transport)) as client, \
             patch.object(config, "DEEPSEEK_API_KEY", "local-test-placeholder"), \
             patch.object(config, "DEEPSEEK_BASE_URL", "https://local-test.invalid/v1"):
            model = create_chat_model(model="local-test", http_client=client)
            decision = invoke_structured(model, TriageDecision, [HumanMessage(content="你好")])
            reply = model.invoke([HumanMessage(content="你好")])
            model.bind_tools([{"type": "function", "function": {"name": "lookup",
                "description": "测试工具", "parameters": {"type": "object", "properties": {}}}}]) \
                .invoke([HumanMessage(content="查订单")])
        self.assertIsInstance(decision, TriageDecision)
        self.assertIn("你好", reply.content)
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", requests[1])
        self.assertNotIn("response_format", requests[2])
        self.assertIn("tools", requests[2])
        self.assertIn("route", requests[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
