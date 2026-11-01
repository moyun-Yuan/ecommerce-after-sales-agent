"""可重复运行的数据初始化脚本。

使用 ``python -m ecommerce_after_sales_agent.seed`` 写入演示客户、订单和工单，并重建
Chroma 政策索引。操作日志、审批和回复等运行数据不会被清空。缺少向量库依赖时，
SQLite 数据仍会完成初始化，并输出跳过索引构建的提示。
"""
from __future__ import annotations

import argparse

from . import config, db

# --- 演示数据 ----------------------------------------------------------------

CUSTOMERS = [
    # 客户编号、姓名、邮箱
    ("CUST-1001", "林晓", ""),
    ("CUST-1002", "张伟", ""),
    ("CUST-1003", "王芳", ""),
    ("CUST-1004", "李明", ""),
    ("CUST-1005", "陈静", ""),
]

CUSTOMER_CONTACTS = {
    "CUST-1001": ("138****1001", "北京市"),
    "CUST-1002": ("139****1002", "杭州市"),
    "CUST-1003": ("137****1003", "上海市"),
    "CUST-1004": ("136****1004", "深圳市"),
    "CUST-1005": ("135****1005", "成都市"),
}

# 客户编号 -> 客户备注列表
STORED_FACTS = {
    "CUST-1001": ["优先使用站内消息联系。", "三年老客户，累计购买20余次。"],
    "CUST-1002": ["曾咨询快递延误问题。"],
    "CUST-1003": ["需要电子发票。"],
    "CUST-1004": ["希望优先选择顺丰配送。"],
    "CUST-1005": ["企业采购客户，需要开票。"],
}

# 工单编号、客户编号、主题、状态
TICKETS = [
    ("TCK-9001", "CUST-1001", "咨询订单物流", "resolved"),
    ("TCK-9002", "CUST-1002", "咨询快递延误", "resolved"),
    ("TCK-9003", "CUST-1003", "咨询退货流程", "resolved"),
    ("TCK-9004", "CUST-1004", "咨询耳机保修", "open"),
    ("TCK-9005", "CUST-1005", "企业采购开票咨询", "open"),
]

# 订单号、客户编号、商品、金额、状态、承运商、运单号、最新轨迹
ORDERS = [
    # 用于演示丢件退款流程的订单。
    ("ORD-5001", "CUST-1001", "无线头戴耳机", 299.00, "lost",
     "顺丰速运", "SF-DEMO-5001", "北京转运中心核实快件遗失，承运商已确认丢件"),
    ("ORD-5002", "CUST-1002", "机械键盘", 199.00, "delivered",
     "中通快递", "ZTO-DEMO-5002", "已送达杭州收件地址，由收件人签收"),
    ("ORD-5003", "CUST-1003", "USB-C 数据线（两条装）", 29.90, "in_transit",
     "圆通速递", "YTO-DEMO-5003", "快件已到达上海转运中心，预计两天内送达"),
    ("ORD-5004", "CUST-1004", "降噪耳塞", 399.00, "delivered",
     "京东物流", "JD-DEMO-5004", "已送达深圳收件地址，由收件人签收"),
    ("ORD-5005", "CUST-1005", "升降办公桌", 1299.00, "processing",
     None, None, "成都仓正在备货，预计48小时内发货"),
]


def seed_sqlite() -> None:
    """清空并重新写入由初始化脚本负责的演示表。"""
    db.init_db()
    with db.get_conn() as conn:
        # 先删除子表，再删除父表，满足外键约束。
        for table in ("refunds", "stored_facts", "tickets", "orders", "customers"):
            conn.execute(f"DELETE FROM {table}")

        conn.executemany(
            "INSERT INTO customers (id, name, email, phone, city, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [(cid, name, email, *CUSTOMER_CONTACTS[cid], db.now()) for cid, name, email in CUSTOMERS],
        )
        conn.executemany(
            "INSERT INTO tickets (id, customer_id, subject, status, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            [(tid, cid, subj, status, db.now()) for tid, cid, subj, status in TICKETS],
        )
        conn.executemany(
            "INSERT INTO orders (id, customer_id, item, amount, status, carrier,"
            " tracking_number, last_scan, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(*row, db.now()) for row in ORDERS],
        )
        facts = [
            (cid, fact, db.now())
            for cid, fact_list in STORED_FACTS.items()
            for fact in fact_list
        ]
        conn.executemany(
            "INSERT INTO stored_facts (customer_id, fact, created_at) VALUES (?, ?, ?)",
            facts,
        )
    print(
        f"  SQLite：{len(CUSTOMERS)}位演示客户、{len(TICKETS)}个工单、"
        f"{len(ORDERS)}笔人民币订单（含1笔已确认丢件）、{sum(len(f) for f in STORED_FACTS.values())}条客户备注。"
    )


def seed_chroma() -> None:
    """为政策文档生成向量并写入持久化 Chroma 集合。"""
    try:
        from .rag import build_index
    except Exception as exc:  # pragma: no cover - only if RAG deps are missing
        print(f"  Chroma: skipped (RAG not available: {exc}).")
        return
    n = build_index()
    print(f"  Chroma: indexed {n} chunks into collection '{config.CHROMA_COLLECTION}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化演示数据或单独更新政策索引。")
    parser.add_argument("--index-only", action="store_true", help="仅重建政策索引，不重置订单与退款。")
    args = parser.parse_args()
    print("正在初始化电商售后演示数据…")
    if not args.index_only:
        seed_sqlite()
    seed_chroma()
    print("初始化完成。")


if __name__ == "__main__":
    main()
