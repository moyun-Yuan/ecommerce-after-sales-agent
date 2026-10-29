"""思考模式与用量统计回归测试；只使用本地 HTTP 传输替身。"""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ecommerce_after_sales_agent import config
from ecommerce_after_sales_agent.llm import create_chat_model, invoke_structured
from ecommerce_after_sales_agent.schemas import TriageDecision
from ecommerce_after_sales_agent.usage import TokenUsageTracker


class ThinkingAndUsageTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.key = patch.object(config, "DEEPSEEK_API_KEY", "local-test-placeholder")
        self.key.start()
        self.endpoint = patch.object(config, "DEEPSEEK_BASE_URL", "https://local-test.invalid/v1")
        self.endpoint.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.key.stop)
        self.addCleanup(self.endpoint.stop)

    @staticmethod
    def response(message, *, prompt=10, completion=7, reasoning=3, cached=4):
        return httpx.Response(200, json={
            "id": "local-test", "object": "chat.completion", "created": 0, "model": "local-test",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion, "prompt_cache_hit_tokens": cached,
                      "completion_tokens_details": {"reasoning_tokens": reasoning}},
        })

    def test_role_defaults_and_independent_override(self):
        for role in ("general", "triage", "solver", "reviewer", "judge"):
            with self.subTest(role=role):
                self.assertFalse(config.model_options(role)["thinking"])
        with patch.dict(os.environ, {"DEEPSEEK_THINKING": "false", "SOLVER_THINKING": "true"}):
            self.assertTrue(config.model_options("solver")["thinking"])
            self.assertFalse(config.model_options("reviewer")["thinking"])
        with patch.dict(os.environ, {"DEEPSEEK_THINKING": "true", "REVIEWER_THINKING": "false"}):
            self.assertTrue(config.model_options("triage")["thinking"])
            self.assertTrue(config.model_options("solver")["thinking"])
            self.assertFalse(config.model_options("reviewer")["thinking"])

    def test_invalid_configuration_fails_early(self):
        for settings in ({"SOLVER_THINKING": "typo"}, {"DEEPSEEK_REASONING_EFFORT": "typo"},
                         {"THINKING_MAX_TOKENS": "0"}):
            with self.subTest(settings=settings), patch.dict(os.environ, settings), self.assertRaises(ValueError):
                config.model_options("solver")

    def test_thinking_tool_round_trip_and_checkpoint_keep_reasoning(self):
        os.environ["SOLVER_THINKING"] = "true"
        requests = []
        tracker = TokenUsageTracker()
        messages = [HumanMessage(content="请查询演示订单")]

        def transport(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(payload["thinking"], {"type": "enabled"})
            self.assertEqual(payload["reasoning_effort"], "low")
            self.assertNotIn("temperature", payload)
            self.assertNotIn(payload.get("tool_choice"), ("required",))
            if len(requests) == 1:
                return self.response({"role": "assistant", "content": "", "reasoning_content": "演示推理一",
                    "tool_calls": [{"id": "call-1", "type": "function", "function": {
                        "name": "lookup", "arguments": json.dumps({"order_id": "ORD-5001"})}}]})
            return self.response({"role": "assistant", "content": "查询完成", "reasoning_content": "演示推理二"})

        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            model = create_chat_model(model="local-test", role="solver", http_client=client).bind_tools([
                {"type": "function", "function": {"name": "lookup", "description": "查询演示订单",
                 "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                                "required": ["order_id"]}}}])
            first = model.invoke(messages, config={"callbacks": [tracker]})
            self.assertEqual(first.additional_kwargs["reasoning_content"], "演示推理一")
            # 使用与检查点一致的序列化器，确认 AIMessage 附加字段不会丢失。
            serializer = JsonPlusSerializer()
            first = serializer.loads_typed(serializer.dumps_typed(first))
            messages.extend([first, ToolMessage(content="订单已找到", tool_call_id="call-1")])
            second = model.invoke(messages, config={"callbacks": [tracker]})
            messages.extend([second, HumanMessage(content="谢谢")])
            model.invoke(messages, config={"callbacks": [tracker]})
        replay = [m for m in requests[2]["messages"] if m["role"] == "assistant"]
        self.assertEqual([m["reasoning_content"] for m in replay], ["演示推理一", "演示推理二"])
        totals = tracker.summary()["totals"]
        self.assertEqual(totals["calls"], 3)
        self.assertEqual(totals["input_tokens"], 30)
        self.assertEqual(totals["output_tokens"], 21)
        self.assertEqual(totals["total_tokens"], 51)
        self.assertEqual(totals["reasoning_tokens"], 9)  # 已包含在 21 个输出内，不再加到 51。
        self.assertEqual(totals["cache_read_tokens"], 12)
        self.assertIn("solver:enabled", tracker.summary()["by_stage"])
        self.assertNotIn("演示推理", json.dumps(tracker.summary(), ensure_ascii=False))

    def test_missing_reasoning_is_rejected_before_http(self):
        os.environ["SOLVER_THINKING"] = "true"
        def transport(request):
            self.fail("缺少思考字段时不应发出请求")
        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            model = create_chat_model(role="solver", http_client=client).bind_tools([
                {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}])
            with self.assertRaisesRegex(ValueError, "reasoning_content"):
                model.invoke([HumanMessage(content="演示"), AIMessage(content="旧的最终回复")])

    def test_thinking_json_output_and_usage_are_compatible(self):
        os.environ["REVIEWER_THINKING"] = "true"
        tracker = TokenUsageTracker()
        requests = []

        def transport(request):
            payload = json.loads(request.content)
            requests.append(payload)
            return self.response({"role": "assistant", "reasoning_content": "演示分析",
                "content": '{"route":"use_tools","reason":"订单查询"}'})

        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            model = create_chat_model(role="reviewer", http_client=client)
            # 回调配置绑定到本次模型，结构化包装之后仍能获得原始用量事件。
            model.callbacks = [tracker]
            result = invoke_structured(model, TriageDecision, [HumanMessage(content="演示订单查询")])
        self.assertEqual(result.route, "use_tools")
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", requests[0])
        self.assertEqual(tracker.total_tokens(), 17)

    def test_disabled_mode_does_not_require_reasoning_history(self):
        requests = []
        def transport(request):
            requests.append(json.loads(request.content))
            return self.response({"role": "assistant", "content": "你好"}, reasoning=0)
        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            for role in ("general", "triage", "solver", "reviewer", "judge"):
                model = create_chat_model(role=role, http_client=client)
                model.invoke([HumanMessage(content="你好"), AIMessage(content="你好")])
        self.assertEqual(len(requests), 5)
        for request in requests:
            self.assertEqual(request["thinking"], {"type": "disabled"})
            self.assertNotIn("reasoning_effort", request)

    def test_missing_usage_is_unknown_not_zero(self):
        tracker = TokenUsageTracker()
        run_id = uuid4()
        tracker.on_chat_model_start({}, [], run_id=run_id, tags=["stage:solver", "thinking:enabled"])
        tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="演示"))]]), run_id=run_id)
        self.assertIsNone(tracker.total_tokens())
        self.assertEqual(tracker.summary()["totals"]["missing_usage_calls"], 1)
        self.assertIsNone(tracker.cost({}))

    def test_missing_reasoning_usage_stays_unknown(self):
        tracker = TokenUsageTracker()
        run_id = uuid4()
        message = AIMessage(content="演示", usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})
        tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
        self.assertEqual(tracker.total_tokens(), 5)
        self.assertIsNone(tracker.summary()["totals"]["reasoning_tokens"])

    def test_failed_calls_do_not_fabricate_usage_or_capture_errors(self):
        tracker = TokenUsageTracker()
        run_id = uuid4()
        tracker.on_chat_model_start({}, [], run_id=run_id, tags=["stage:solver", "thinking:enabled"])
        tracker.on_llm_error(RuntimeError("不应保存的错误正文"), run_id=run_id)
        summary = tracker.summary()
        self.assertEqual(summary["totals"]["failed_calls"], 1)
        self.assertEqual(summary["totals"]["missing_usage_calls"], 1)
        self.assertIsNone(summary["totals"]["total_tokens"])
        self.assertNotIn("不应保存", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
