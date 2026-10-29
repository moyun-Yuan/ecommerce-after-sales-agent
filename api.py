"""售后智控 HTTP API。

客户接口负责提交咨询和查询自己的处理结果；员工接口使用独立令牌保护，
用于读取待审批申请和提交审批决定。接口只包装现有工作流，不复制业务逻辑。
"""
from __future__ import annotations

import hmac
import threading
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from ecommerce_after_sales_agent import config, db, graph


class ChatRequest(BaseModel):
    """客户发起一次咨询所需的数据。"""

    customer_id: str = Field(min_length=1, max_length=50)
    query: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=100)


class ApprovalDecision(BaseModel):
    """员工对敏感操作的审批决定。"""

    approved: bool
    staff_id: str = Field(min_length=1, max_length=50)


@lru_cache(maxsize=1)
def _workflow():
    """在单进程内复用工作流和 SQLite 检查点连接。"""
    return graph.get_app()


_workflow_lock = threading.RLock()


def _require_staff_token(x_staff_token: str | None = Header(default=None)) -> None:
    """校验员工接口令牌；未配置令牌时直接关闭员工 API。"""
    expected = config.STAFF_API_TOKEN
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务端尚未配置 STAFF_API_TOKEN。",
        )
    if not x_staff_token or not hmac.compare_digest(x_staff_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="员工令牌无效。",
        )


def _customer_exists(customer_id: str) -> bool:
    """确认演示客户存在。"""
    with db.get_conn() as conn:
        return conn.execute(
            "SELECT 1 FROM customers WHERE id = ?", (customer_id,)
        ).fetchone() is not None


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    """仅返回可序列化且适合接口展示的状态字段。"""
    return {
        "route": state.get("route"),
        "approval_decision": state.get("approval_decision"),
        "final_reply": state.get("final_reply"),
        "citations": state.get("citations", []),
        "escalated": bool(state.get("escalated")),
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时创建数据表；演示数据仍由 seed 命令显式初始化。"""
    db.init_db()
    yield


app = FastAPI(
    title="售后智控 API",
    version="0.2.0",
    description="电商售后 Agent 的客户咨询与人工审批接口。",
    lifespan=lifespan,
)


@app.get("/health", tags=["系统"])
def health() -> dict[str, Any]:
    """轻量健康检查，不调用大模型。"""
    return {
        "status": "ok",
        "model_configured": bool(config.DEEPSEEK_API_KEY),
        "data_profile": config.DATA_PROFILE,
    }


@app.post("/api/v1/chat", tags=["客户"])
def chat(request: ChatRequest) -> dict[str, Any]:
    """提交客户问题；遇到敏感操作时返回待审批信息。"""
    if not _customer_exists(request.customer_id):
        raise HTTPException(status_code=404, detail="客户不存在。")
    thread_id = request.thread_id or f"api-{uuid.uuid4().hex[:12]}"
    with _workflow_lock:
        result = graph.run_turn(
            request.query,
            request.customer_id,
            thread_id,
            app=_workflow(),
        )
    response: dict[str, Any] = {"status": result["status"], "thread_id": thread_id, "usage": result["usage"]}
    if result["status"] == "interrupted":
        response["approval"] = result["interrupt"]
    else:
        response.update(_public_state(result["state"]))
    return response


@app.get("/api/v1/threads/{thread_id}", tags=["客户"])
def thread_result(
    thread_id: str,
    customer_id: str = Query(min_length=1, max_length=50),
) -> dict[str, Any]:
    """查询指定客户自己的会话状态和最终回复。"""
    with _workflow_lock:
        state_data = graph.peek_state(thread_id, app=_workflow())
    if not state_data:
        raise HTTPException(status_code=404, detail="会话不存在。")
    if state_data.get("customer_id") != customer_id:
        raise HTTPException(status_code=403, detail="无权查看该会话。")
    return {"thread_id": thread_id, **_public_state(state_data)}


@app.get(
    "/api/v1/staff/approvals",
    tags=["员工"],
    dependencies=[Depends(_require_staff_token)],
)
def pending_approvals() -> dict[str, Any]:
    """列出待员工处理的敏感操作。"""
    items = db.get_pending_approvals()
    return {"count": len(items), "items": items}


@app.post(
    "/api/v1/staff/approvals/{approval_id}/decision",
    tags=["员工"],
    dependencies=[Depends(_require_staff_token)],
)
def decide(approval_id: int, decision: ApprovalDecision) -> dict[str, Any]:
    """提交审批决定并恢复对应的 LangGraph 任务。"""
    approval = db.get_approval(approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="审批记录不存在。")
    if approval["status"] != "pending":
        raise HTTPException(status_code=409, detail="该审批已经处理，不能重复提交。")
    with _workflow_lock:
        result = graph.resume_turn(
            approval["thread_id"],
            approved=decision.approved,
            decided_by=decision.staff_id,
            app=_workflow(),
        )
    return {
        "approval_id": approval_id,
        "thread_id": approval["thread_id"],
        "status": "approved" if decision.approved else "rejected",
        "usage": result["usage"],
        **_public_state(result["state"]),
    }
