"""处理节点的框架 Agent；中间件限制工具权限、调用预算并恢复业务上下文。"""
from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from . import tools

LIMIT_REPLY = "本轮自动处理已达到调用上限，请补充信息或联系人工继续处理；未批准的退款不会执行。"


class SolverModelLimit(ModelCallLimitMiddleware):
    """复用框架计数和终止机制，将达到上限时的提示改为客户可读的说明。"""

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        update = super().before_model(state, runtime)
        if update and update.get("jump_to") == "end":
            update["messages"] = [AIMessage(content=LIMIT_REPLY)]
        return update


class SolverToolLimit(ToolCallLimitMiddleware):
    """限制一次处理中的工具请求数，避免单次模型响应生成过大的工具批次。"""

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        update = super().after_model(state, runtime)
        if update and update.get("jump_to") == "end":
            update["messages"][-1] = AIMessage(content=LIMIT_REPLY)
        return update


class SolverToolGuard(AgentMiddleware):
    """工具清单之外再做执行层校验；模型不能靠编造工具名越过权限限制。"""

    def __init__(self, active_tools, thread_id, customer_id):
        self.allowed_names = {item.name for item in active_tools}
        self.thread_id = thread_id
        self.customer_id = customer_id

    @staticmethod
    def _reject(call, reason):
        return ToolMessage(
            content=json.dumps({"status": "rejected", "error": reason}, ensure_ascii=False),
            tool_call_id=call["id"], name=call["name"], status="error",
        )

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] not in self.allowed_names:
            return self._reject(call, "本轮不允许调用该工具。")
        # 工具由框架的工具节点调度；在实际执行位置明确恢复归属和审计上下文。
        tools.set_run_context(self.thread_id, self.customer_id)
        try:
            response = handler(request)
        except ValidationError:
            return self._reject(call, "工具参数格式无效，请检查参数类型。")
        if isinstance(response, ToolMessage) and response.status == "error":
            return self._reject(call, "工具参数格式无效，请检查参数类型。")
        return response


def invoke_solver_agent(model, active_tools, messages, *, thread_id, customer_id,
                        max_model_calls=5, max_tool_calls=20):
    """由 create_agent 管理模型/工具循环，返回本次新增的原始消息。

    内层不持久化对话，避免与外层会话状态重复保存；人工中断仍由外层 approval
    节点负责。串行调度工具，保留先查订单、再操作等顺序及原有 SQLite 使用方式。
    """
    agent = create_agent(
        model=model,
        tools=active_tools,
        middleware=[
            SolverModelLimit(run_limit=max_model_calls, exit_behavior="end"),
            SolverToolLimit(run_limit=max_tool_calls, exit_behavior="end"),
            SolverToolGuard(active_tools, thread_id, customer_id),
        ],
        checkpointer=False,
        name="after_sales_solver",
    )
    # 内层中间件也占图步骤；业务预算由上面的计数器控制，而非依赖递归报错中断。
    result = agent.invoke(
        {"messages": messages},
        config={"max_concurrency": 1, "recursion_limit": max_model_calls * 8 + 10},
    )
    return result["messages"][len(messages):]
