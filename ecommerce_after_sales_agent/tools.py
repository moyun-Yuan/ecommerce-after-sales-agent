"""Agent 可调用的业务工具和由流程控制的敏感工具。

订单、物流和 CRM 使用 SQLite 模拟，政策查询使用真实的向量检索与重排。
所有调用都会写入操作日志；退款执行不暴露给模型，只能由审批节点调用。
"""
from __future__ import annotations

import contextvars
import json

from langchain_core.tools import tool

from . import db, rag
from .i18n import localize_order

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
    with db.get_conn() as conn:
        return conn.execute(
            "SELECT * FROM orders WHERE id = ? AND customer_id = ?",
            (order_id, customer_id),
        ).fetchone()

@tool
def get_order(order_id: str) -> str:
    """查询当前客户名下订单，返回商品、金额和状态。"""
    row = _get_owned_order(order_id)
    result = localize_order(dict(row)) if row else {"error": "未找到该订单或当前客户无权查看。"}
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
def create_refund_draft(order_id: str, amount: float) -> str:
    """校验订单与金额后创建退款草案，不执行退款。"""
    row = _get_owned_order(order_id)
    if not row:
        result = {"status": "rejected", "error": "未找到该订单或当前客户无权操作。"}
    elif amount <= 0 or round(float(amount), 2) != round(float(row["amount"]), 2):
        result = {
            "status": "rejected",
            "error": "退款金额必须等于该订单的实付金额。",
            "paid_amount": float(row["amount"]),
        }
    else:
        result = {
        "draft_id": f"DRAFT-{order_id}",
        "order_id": order_id,
        "amount": round(float(amount), 2),
        "currency": "CNY",
        "status": "drafted",
        "note": "退款草案已提交，等待员工审批；尚未执行退款。",
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
) -> dict:
    """审批后执行模拟退款，并通过数据库唯一约束防止重复退款。"""
    with db.get_conn() as conn:
        # 提前取得写锁，避免界面与 API 进程同时为同一订单创建退款。
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT customer_id, amount FROM orders WHERE id = ?",
            (order_id,),
        ).fetchone()
        if not row or (customer_id and row["customer_id"] != customer_id):
            raise ValueError("订单不存在或不属于当前客户。")
        paid_amount = round(float(row["amount"]), 2)
        refund_amount = round(float(amount), 2)
        if refund_amount <= 0 or refund_amount != paid_amount:
            raise ValueError("退款金额必须等于订单实付金额。")

        existing = conn.execute(
            "SELECT id, amount, currency, status FROM refunds WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        if existing:
            result = {
                "refund_id": existing["id"],
                "order_id": order_id,
                "amount": float(existing["amount"]),
                "currency": existing["currency"],
                "status": "already_refunded",
            }
            action_name = "duplicate_refund_blocked"
        else:
            result = {
                "refund_id": f"RF-{order_id}",
                "order_id": order_id,
                "amount": refund_amount,
                "currency": "CNY",
                "status": "refunded",
            }
            conn.execute(
                "INSERT INTO refunds (id, order_id, customer_id, amount, currency, status, thread_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (result["refund_id"], order_id, row["customer_id"], refund_amount,
                 "CNY", "refunded", thread_id, db.now()),
            )
            action_name = "issue_refund"
    db.log_action(
        action_name, {"order_id": order_id, "amount": amount}, result,
        thread_id=thread_id, customer_id=row["customer_id"],
    )
    return result


def send_reply(text: str, thread_id: str | None = None, customer_id: str | None = None) -> dict:
    """记录确定性的发送动作；演示界面随后展示同一条回复。"""
    result = {"status": "sent", "chars": len(text)}
    db.log_action("send_reply", {"text": text}, result, thread_id=thread_id, customer_id=customer_id)
    return result
