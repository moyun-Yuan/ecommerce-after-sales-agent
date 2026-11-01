"""SQLite 业务数据表和轻量数据访问函数。

业务库保存客户、订单、审批、退款结果和操作日志；LangGraph 的运行状态使用
另一份 checkpoints.sqlite 保存，两类数据相互独立。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any

from . import config
from .money import from_cents, to_cents

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    phone       TEXT NOT NULL DEFAULT '',
    city        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id          TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    subject     TEXT NOT NULL,
    status      TEXT NOT NULL,        -- 工单状态：open | resolved | escalated
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stored_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    fact        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(id),
    item            TEXT NOT NULL,
    amount          REAL NOT NULL,
    status          TEXT NOT NULL,    -- 物流状态：delivered | in_transit | lost | processing
    carrier         TEXT,
    tracking_number TEXT,
    last_scan       TEXT,             -- 最近一次物流轨迹说明
    created_at      TEXT NOT NULL
);

-- 操作审计：记录工具调用和流程执行动作。
CREATE TABLE IF NOT EXISTS actions_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT,
    customer_id TEXT,
    action      TEXT NOT NULL,
    payload     TEXT,                 -- JSON 输入
    result      TEXT,                 -- JSON 结果
    created_at  TEXT NOT NULL
);

-- 人工审批：控制退款等敏感操作。
CREATE TABLE IF NOT EXISTS approvals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    action      TEXT NOT NULL,
    payload     TEXT,                 -- JSON 操作方案
    status      TEXT NOT NULL,        -- 审批状态：pending | approved | rejected | failed
    decided_by  TEXT,
    created_at  TEXT NOT NULL,
    decided_at  TEXT
);

-- 最终答复：由 respond 节点保存。
CREATE TABLE IF NOT EXISTS responses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    customer_id TEXT,
    ticket_id   TEXT,
    reply       TEXT NOT NULL,
    citations   TEXT,                 -- JSON 来源编号列表
    created_at  TEXT NOT NULL
);

-- 一次退款申请只执行一次；同一订单可有多笔部分退款。
CREATE TABLE IF NOT EXISTS refunds (
    id          TEXT PRIMARY KEY,
    request_id  TEXT NOT NULL UNIQUE,
    order_id    TEXT NOT NULL REFERENCES orders(id),
    customer_id TEXT NOT NULL REFERENCES customers(id),
    amount      REAL NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    currency    TEXT NOT NULL,
    status      TEXT NOT NULL,
    thread_id   TEXT,
    created_at  TEXT NOT NULL
);
"""


def now() -> str:
    """生成带 UTC+08:00 偏移量的北京时间。"""
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def get_conn() -> sqlite3.Connection:
    """打开 SQLite 连接，启用按列名访问和外键检查。"""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """创建表并兼容迁移旧退款记录，不清空已有订单或退款。"""
    with closing(get_conn()) as conn:
        conn.executescript(SCHEMA)
        _migrate_refunds(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_refunds_order ON refunds(order_id)")
        conn.commit()


def _migrate_refunds(conn: sqlite3.Connection) -> None:
    """旧版按订单唯一；迁移前备份，结构替换和数据复制在同一事务中完成。"""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(refunds)")}
    if {"request_id", "amount_cents"} <= columns:
        return
    backup = config.DB_PATH.with_name(
        f"{config.DB_PATH.stem}.before-partial-refunds-{uuid.uuid4().hex[:8]}.sqlite"
    )
    with closing(sqlite3.connect(backup)) as destination:
        conn.backup(destination)
    conn.execute("BEGIN IMMEDIATE")
    try:
        # 另一个进程可能已完成迁移，取得写锁后重新核对。
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(refunds)")}
        if {"request_id", "amount_cents"} <= columns:
            conn.commit()
            return
        conn.execute("""CREATE TABLE refunds_migration (
            id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE,
            order_id TEXT NOT NULL REFERENCES orders(id),
            customer_id TEXT NOT NULL REFERENCES customers(id),
            amount REAL NOT NULL, amount_cents INTEGER NOT NULL CHECK(amount_cents > 0),
            currency TEXT NOT NULL, status TEXT NOT NULL, thread_id TEXT,
            created_at TEXT NOT NULL
        )""")
        for row in conn.execute("SELECT * FROM refunds").fetchall():
            conn.execute(
                "INSERT INTO refunds_migration VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row["id"], f"legacy:{row['id']}", row["order_id"], row["customer_id"],
                 row["amount"], to_cents(row["amount"]), row["currency"], row["status"],
                 row["thread_id"], row["created_at"]),
            )
        conn.execute("DROP TABLE refunds")
        conn.execute("ALTER TABLE refunds_migration RENAME TO refunds")
        if conn.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("退款表迁移外键检查失败，已保留原表与备份。")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def refund_balance(conn: sqlite3.Connection, order_id: str, paid_amount) -> dict:
    """读取成功退款累计；调用方负责在执行退款时持有写事务。"""
    paid = to_cents(paid_amount)
    refunded = int(conn.execute(
        "SELECT COALESCE(SUM(amount_cents), 0) FROM refunds "
        "WHERE order_id = ? AND status = 'refunded'", (order_id,),
    ).fetchone()[0])
    if refunded > paid:
        raise ValueError("历史累计退款超过实付金额，请人工核对。")
    return {
        "paid_amount": from_cents(paid), "refunded_amount": from_cents(refunded),
        "remaining_amount": from_cents(paid - refunded), "remaining_cents": paid - refunded,
    }


# --- 工具和节点使用的数据写入函数 -------------------------------------------

def log_action(
    action: str,
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    thread_id: str | None = None,
    customer_id: str | None = None,
) -> None:
    """追加一条操作日志，用于追踪工具调用和敏感操作。"""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO actions_log (thread_id, customer_id, action, payload, result, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                thread_id,
                customer_id,
                action,
                json.dumps(payload) if payload is not None else None,
                json.dumps(result) if result is not None else None,
                now(),
            ),
        )


def record_approval(thread_id: str, action: str, payload: dict[str, Any]) -> int:
    """取得或创建待审批记录，并返回审批编号。

    LangGraph 恢复中断节点时会再次执行 interrupt() 之前的代码，因此这里必须
    复用同一会话的待审批记录，避免产生重复的“待处理”数据。
    """
    with closing(get_conn()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT id, status, payload FROM approvals WHERE thread_id = ? AND action = ?"
            " ORDER BY id DESC",
            (thread_id, action),
        ).fetchall()
        for row in existing:
            previous = json.loads(row["payload"]) if row["payload"] else {}
            same_request = (previous.get("request_id") == payload["request_id"]
                            if payload.get("request_id") else previous == payload)
            if same_request:
                return int(row["id"])
            if row["status"] == "pending":
                raise ValueError("当前会话已有另一笔待审批申请，请先处理。")
        cur = conn.execute(
            "INSERT INTO approvals (thread_id, action, payload, status, created_at)"
            " VALUES (?, ?, ?, 'pending', ?)",
            (thread_id, action, json.dumps(payload), now()),
        )
        return int(cur.lastrowid)


def get_pending_approvals() -> list[dict[str, Any]]:
    """返回全部待处理审批，供员工工作台和 API 使用。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, thread_id, action, payload, created_at FROM approvals"
            " WHERE status = 'pending' ORDER BY id"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
        out.append(d)
    return out


def latest_response_for_thread(thread_id: str) -> dict[str, Any] | None:
    """读取某个会话最近一次保存的最终回复。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT reply, citations, created_at FROM responses WHERE thread_id = ?"
            " ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    return dict(row) if row else None


def get_approval(approval_id: int) -> dict[str, Any] | None:
    """按审批编号读取一条审批记录，并解析其中的 JSON 载荷。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, thread_id, action, payload, status, decided_by, created_at, decided_at "
            "FROM approvals WHERE id = ?",
            (approval_id,),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["payload"] = json.loads(result["payload"]) if result["payload"] else {}
    return result


def decide_approval(approval_id: int, approved: bool, decided_by: str = "human", *,
                    approved_amount: float | None = None, execution_error: str | None = None) -> bool:
    """只处理仍为 pending 的审批；返回本次是否成功更新。"""
    with closing(get_conn()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT payload FROM approvals WHERE id = ? AND status = 'pending'", (approval_id,),
        ).fetchone()
        if not row:
            return False
        payload = json.loads(row["payload"]) if row["payload"] else {}
        requested = payload.get("amount")
        if approved_amount is not None:
            payload.setdefault("requested_amount", requested)
            payload["approved_amount"] = approved_amount
            payload["amount"] = approved_amount
        if execution_error:
            payload["execution_error"] = execution_error
        decision = "failed" if execution_error else ("approved" if approved else "rejected")
        cur = conn.execute(
            "UPDATE approvals SET status = ?, decided_by = ?, decided_at = ?, payload = ? "
            "WHERE id = ? AND status = 'pending'",
            (decision, decided_by, now(), json.dumps(payload), approval_id),
        )
        if approved_amount is not None and requested is not None and to_cents(requested) != to_cents(approved_amount):
            conn.execute(
                "INSERT INTO actions_log (thread_id, action, payload, result, created_at) "
                "SELECT thread_id, 'approval_amount_adjusted', ?, ?, ? FROM approvals WHERE id = ?",
                (json.dumps({"requested_amount": requested, "approved_amount": approved_amount,
                             "decided_by": decided_by}),
                 json.dumps({"approval_id": approval_id, "status": decision}), now(), approval_id),
            )
        return cur.rowcount == 1


def save_response(
    thread_id: str,
    reply: str,
    citations: list[str],
    customer_id: str | None = None,
    ticket_id: str | None = None,
) -> None:
    """保存最终对客回复及其政策引用。"""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO responses (thread_id, customer_id, ticket_id, reply, citations, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, customer_id, ticket_id, reply, json.dumps(citations), now()),
        )
