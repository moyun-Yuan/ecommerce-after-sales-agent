"""用虚构订单检查普通回复、工具往返及 Token；一次模式三次真实调用，会产生费用。

--compare-thinking 对同一模型分别检查开关，共六次调用，不是业务质量评测。
只将数字用量报告写入 .cache，不保存提示词、回复正文或思考正文。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from ecommerce_after_sales_agent.llm import create_chat_model
from ecommerce_after_sales_agent.usage import TokenUsageTracker


@tool
def echo_order_id(order_id: str) -> str:
    """回显订单号，仅检查连接，不读取或修改业务订单。"""
    return json.dumps({"order_id": order_id, "connection_check": "ok"})


def run_check(role: str) -> dict:
    tracker = TokenUsageTracker()
    run_config = {"callbacks": [tracker]}
    model = create_chat_model(role=role)
    reply = model.invoke("请只回复：连接成功", config=run_config)
    print("普通调用:", reply.content)
    print("Token 用量:", reply.usage_metadata)

    messages = [HumanMessage(content="请调用 echo_order_id 检查订单编号 ORD-CHECK-001。")]
    # 思考模式不支持 required 或强制选择某个工具，使用 auto 检查自然工具选择。
    with_tool = model.bind_tools([echo_order_id], tool_choice="auto")
    proposal = with_tool.invoke(messages, config=run_config)
    if not proposal.tool_calls:
        raise RuntimeError("普通调用成功，但未收到工具调用。请把此错误反馈给老师。")
    messages.append(proposal)
    for call in proposal.tool_calls:
        if call["name"] != echo_order_id.name:
            raise RuntimeError("模型返回了检查范围外的工具名称。")
        print("工具调用:", call["name"], call["args"])
        result = echo_order_id.invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    final = with_tool.invoke(messages, config=run_config)
    if final.tool_calls:
        raise RuntimeError("工具返回后模型仍请求额外调用，本次三步检查未通过。")
    print("工具结果回复:", final.content)
    print("模型连接与工具调用检查通过。")
    report = tracker.summary()
    print("本次模式用量：", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="真实接口小检查，产生 API 费用，不加载 RAG。")
    parser.add_argument("--role", choices=["general", "triage", "solver", "reviewer", "judge"], default="solver")
    parser.add_argument("--compare-thinking", action="store_true", help="关闭/开启各检查三次，共六次调用")
    args = parser.parse_args()
    key = f"{args.role.upper()}_THINKING"
    previous = os.environ.get(key)
    reports = {}
    try:
        if args.compare_thinking:
            for mode in ("false", "true"):
                os.environ[key] = mode
                print(f"检查角色 {args.role}，思考开关 {mode}")
                reports[mode] = run_check(args.role)
        else:
            reports["configured"] = run_check(args.role)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    path = Path(__file__).resolve().parent / ".cache" / "model-check-usage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print("数字用量报告已保存：", path)


if __name__ == "__main__":
    main()
