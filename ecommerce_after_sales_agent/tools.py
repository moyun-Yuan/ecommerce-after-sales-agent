"""Agent 可调用的业务工具和由流程控制的敏感工具。

订单、物流和 CRM 使用 SQLite 模拟，政策查询使用真实的向量检索与重排。
所有调用都会写入操作日志；退款执行不暴露给模型，只能由审批节点调用。
"""
from __future__ import annotations

import contextvars
import json
import uuid
from contextlib import closing

from langchain_core.tools import tool
from pydantic import StrictFloat

from . import db, rag
from .i18n import localize_order
from .money import from_cents, to_cents

# 当前运行上下文：让工具能够校验客户归属并关联操作日志。
_run_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("run_ctx", default={})


def set_run_context(thread_id: str | None, customer_id: str | None) -> None:
    _run_ctx.set({"thread_id": thread_id, "customer_id": customer_id})


def _ctx() -> dict:
    return _run_ctx.get()


def _log(action: str, payload: dict, result: dict) -> None:
    c = _ctx()
    db.log_action(
        action,
        payload=payload,
        result=result,
        thread_id=c.get("thread_id"),
        customer_id=c.get("customer_id"),
    )


# --- 模拟外部业务工具 --------------------------------------------------------

def _get_owned_order(order_id: str):
    """查询当前客户名下的订单，防止模型跨客户读取订单。"""
    customer_id = _ctx().get("customer_id")
    if not customer_id:
        return None
    with closing(db.get_conn()) as conn:
        return conn.execute(
            "SELECT * FROM orders WHERE id = ? AND customer_id = ?",
            (order_id, customer_id),
        ).fetchone()

@tool
def get_order(order_id: str) -> str:
    """查询当前客户名下订单，返回商品、金额和状态。"""
    row = _get_owned_order(order_id)
    if row:
        result = localize_order(dict(row))
        with closing(db.get_conn()) as conn:
            result.update(db.refund_balance(conn, order_id, row["amount"]))
        result.pop("remaining_cents", None)
    else:
        result = {"error": "未找到该订单或当前客户无权查看。"}
    _log("get_order", {"order_id": order_id}, result)
    return json.dumps(result, ensure_ascii=False)


@tool
def get_tracking(order_id: str) -> str:
    """查询当前客户名下订单的承运商和物流状态。"""
    row = _get_owned_order(order_id)
    if not row:
        result = {"error": "未找到该订单或当前客户无权查看。"}
    else:
        result = localize_order({
            "order_id": row["id"],
            "carrier": row["carrier"],
            "tracking_number": row["tracking_number"],
            "status": row["status"],
            "last_scan": row["last_scan"],
            "is_lost": row["status"] == "lost",
        })
    _log("get_tracking", {"order_id": order_id}, result)
    return json.dumps(result, ensure_ascii=False)


@tool
def check_refund_policy(query: str) -> str:
    """检索政策知识库，返回带来源编号的相关段落。"""
    results = rag.retrieve(query, k=3)
    _log("check_refund_policy", {"query": query}, {"doc_ids": [r["doc_id"] for r in results]})
    return json.dumps({"results": results}, ensure_ascii=False)


@tool
def create_refund_draft(order_id: str, amount: StrictFloat) -> str:
    """按剩余可退金额创建全额或部分退款草案；不预占余额、不执行退款。"""
    row = _get_owned_order(order_id)
    if not row:
        result = {"status": "rejected", "error": "未找到该订单或当前客户无权操作。"}
    else:
        with closing(db.get_conn()) as conn:
            balance = db.refund_balance(conn, order_id, row["amount"])
        remaining = balance.pop("remaining_cents")
        try:
            cents = to_cents(amount)
            if cents <= 0 or cents > remaining:
                raise ValueError("退款金额必须大于零，且不得超过剩余可退金额。")
        except ValueError as exc:
            result = {"status": "rejected", "error": str(exc), **balance}
        else:
            request_id = f"DRAFT-{uuid.uuid4().hex}"
            result = {
                "draft_id": request_id, "request_id": request_id,
                "order_id": order_id, "amount": from_cents(cents),
                "currency": "CNY", "status": "drafted", **balance,
                "note": "退款草案等待员工审批；尚未执行或占用可退余额，执行时再次校验。",
            }
    _log("create_refund_draft", {"order_id": order_id, "amount": amount}, result)
    return json.dumps(result, ensure_ascii=False)


@tool
def log_to_crm(customer_id: str, note: str) -> str:
    """为当前客户写入一条模拟 CRM 备注。"""
    current_customer = _ctx().get("customer_id")
    if not current_customer or customer_id != current_customer:
        result = {"status": "rejected", "error": "不能修改其他客户的 CRM 记录。"}
        _log("log_to_crm", {"customer_id": customer_id, "note": note}, result)
        return json.dumps(result, ensure_ascii=False)
    result = {"customer_id": customer_id, "note": note, "status": "logged"}
    _log("log_to_crm", {"customer_id": customer_id, "note": note}, result)
    return json.dumps(result)


# 提供给模型的工具列表；敏感的 issue_refund 和 send_reply 不在其中。
SOLVER_TOOLS = [get_order, get_tracking, check_refund_policy, create_refund_draft, log_to_crm]
SOLVER_TOOLS_BY_NAME = {t.name: t for t in SOLVER_TOOLS}


# --- 只能由工作流调用的敏感或确定性工具 ------------------------------------

def issue_refund(
    order_id: str,
    amount: float,
    thread_id: str | None = None,
    customer_id: str | None = None,
    *, request_id: str,
) -> dict:
    """按申请编号幂等执行退款，在写事务内校验累计金额，防止并发超退。"""
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
        raise ValueError("退款必须提供有效的唯一申请编号。")
    refund_cents = to_cents(amount)
    if refund_cents <= 0:
        raise ValueError("退款金额必须大于零。")
    with closing(db.get_conn()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT customer_id, amount FROM orders WHERE id = ?",
            (order_id,),
        ).fetchone()
        if not row or not customer_id or row["customer_id"] != customer_id:
            raise ValueError("订单不存在或不属于当前客户。")
        existing = conn.execute(
            "SELECT * FROM refunds WHERE request_id = ?", (request_id,),
        ).fetchone()
        if existing:
            if (existing["order_id"] != order_id or existing["customer_id"] != customer_id
                    or existing["amount_cents"] != refund_cents):
                raise ValueError("同一退款申请编号不能用于不同订单、客户或金额。")
            result = {
                "refund_id": existing["id"], "request_id": request_id,
                "order_id": order_id, "amount": from_cents(refund_cents),
                "currency": existing["currency"], "status": "already_refunded",
            }
            action_name = "duplicate_refund_blocked"
        else:
            balance = db.refund_balance(conn, order_id, row["amount"])
            if refund_cents > balance["remaining_cents"]:
                raise ValueError(f"退款金额超过剩余可退金额 {balance['remaining_amount']:.2f} 元。")
            result = {
                "refund_id": f"RF-{uuid.uuid4().hex}", "request_id": request_id,
                "order_id": order_id, "amount": from_cents(refund_cents),
                "currency": "CNY", "status": "refunded",
            }
            conn.execute(
                "INSERT INTO refunds (id, request_id, order_id, customer_id, amount, amount_cents, "
                "currency, status, thread_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (result["refund_id"], request_id, order_id, customer_id, result["amount"],
                 refund_cents, "CNY", "refunded", thread_id, db.now()),
            )
            action_name = "issue_refund"
        result.update(db.refund_balance(conn, order_id, row["amount"]))
        result.pop("remaining_cents", None)
    db.log_action(
        action_name, {"order_id": order_id, "amount": amount, "request_id": request_id}, result,
        thread_id=thread_id, customer_id=row["customer_id"],
    )
    return result


def send_reply(text: str, thread_id: str | None = None, customer_id: str | None = None) -> dict:
    """记录确定性的发送动作；演示界面随后展示同一条回复。"""
    result = {"status": "sent", "chars": len(text)}
    db.log_action("send_reply", {"text": text}, result, thread_id=thread_id, customer_id=customer_id)
    return result
