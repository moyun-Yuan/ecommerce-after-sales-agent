# 框架 Agent 与业务工作流

`solver` 使用 LangChain 1.4.7 的 `create_agent` 执行模型与工具往返。外层业务流程仍由 LangGraph 编排：

```text
triage → solver（create_agent）→ approval → reviewer → respond
           ↑                                  │
           └────────── 答复需要修正 ────────────┘
```

## 源码职责

- `graph.py:solver_node`：按分流结果选择工具，组织账户背景和对话，调用内层 Agent；随后把消息整理为 `tool_calls`、`retrieved`、`citations`、`proposed_response`、`proposed_action`。
- `solver_agent.py:invoke_solver_agent`：创建和运行框架 Agent。框架负责向模型绑定工具、执行工具、追加 `ToolMessage` 并继续调用模型，项目不再手写这个循环。
- `tools.py`：实际业务工具、订单归属和金额校验、操作审计。
- `graph.py:approval_node`：继续负责保存人工审批、`interrupt()` 暂停及批准后的模拟退款。实际退款工具未交给模型。

`solver_node` 中遍历返回消息的循环只负责提取数据，不调用模型或执行工具，不是原来的工具执行循环。

## 调用方式

核心创建代码为：

```python
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
```

`model` 使用现有模型工厂，按 `role="solver"` 读取模型、思考开关等配置。这里没有设置 `response_format`，所以不会要求客户答复服从分流节点的 JSON/Pydantic 输出契约。

## 权限和业务边界

- `answer_directly` 仅绑定政策查询工具；`use_tools` 才能获得订单、物流、草案等工具。
- `SolverToolGuard` 在执行位置再次检查工具白名单。即使模型编造了未授权的工具名，也只会收到拒绝消息。
- 工具执行时设置当前 `thread_id` 和 `customer_id`，保持订单归属检查和操作日志关联。
- 工具参数校验错误转换为拒绝消息，允许模型修正；不把数据库连接或外部服务的未知异常统一伪装成参数错误。
- 本轮退款已经批准、拒绝或执行失败后，审查重试不再绑定草案工具，执行白名单也同步移除，防止在同一轮创建第二笔退款。
- 同一工具批次串行执行，保留原来先查询、再创建草案等执行顺序。框架化不改变退款余额校验、幂等编号或人工审批边界。

## 预算和持久化

一次 `solver` 调用最多发起 5 次模型调用、20 次工具请求。达到模型预算时停止继续调用；若某批工具请求超出预算，该批工具不执行，框架补齐失败消息并结束。客户收到调用达到上限的提示，而不是递归异常。

审查重试会重新进入 `solver`，拥有新的节点调用预算；这些限制不是整个会话的累计上限，也不包括 SDK 内部网络重试。

内层 Agent 不保存检查点：它的一次完整处理在外层 `solver` 节点内完成，完成后将业务结果交回外层状态。客户历史、最终答复和人工审批暂停位置仍由外层检查点管理。进程在内层执行过程中退出时，不提供从某一次内部模型调用精确续接的能力。

原始 `AIMessage` 保留思考模式工具往返所需的 `reasoning_content`；外层仍只保存最终对话和选定的业务结果，Token 统计不记录思考正文。框架内部调用会继承外层回调。

## 安装与验证

在原项目环境执行：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

新增 10 项框架 Agent 测试，覆盖政策路由限制、伪造工具拒绝、参数校验、模型/工具预算、执行顺序、客户上下文、审查重试和两种思考模式下的模型协议及 Token 回调。加上原有 56 项，共 66 项离线测试。

测试使用临时业务库、聊天模型替身或 `httpx.MockTransport`；没有请求真实模型，没有迁移或重置用户现有业务数据。真实模型的业务质量、费用和响应时间仍需另行实测。历史思考模式对照结果属于改造前版本，不能直接当作当前框架实现的最新线上成绩；实验源码指纹已纳入 `solver_agent.py`，避免混用两版结果。

接口用法参考 [LangChain 官方 Agents 文档](https://docs.langchain.com/oss/python/langchain/agents) 和 [中间件参考](https://reference.langchain.com/python/langchain/agents/middleware)。
