"""业务状态名称和金额显示格式。"""
from __future__ import annotations

STATUS_NAMES = {
    "lost": "运输丢失", "in_transit": "运输中", "delivered": "已送达",
    "processing": "备货中", "drafted": "待审批", "refunded": "已模拟退款",
    "refund_rejected": "退款未获批准",
    "refund_failed": "退款未能执行",
}
ROUTE_NAMES = {
    "answer_directly": "政策问答 / 日常咨询",
    "use_tools": "调用工具处理",
    "escalate_to_human": "转交人工客服",
}
TOOL_NAMES = {
    "get_order": "查询订单", "get_tracking": "查询物流",
    "check_refund_policy": "检索售后政策", "create_refund_draft": "创建退款草案",
    "log_to_crm": "记录客户备注", "issue_refund": "模拟执行退款", "send_reply": "保存回复",
}


def customer_label(customer_id: str, original: str) -> str:
    return original


def status_label(status: str) -> str:
    return STATUS_NAMES.get(status, status)


def localize_order(order: dict) -> dict:
    result = dict(order)
    if "amount" in result:
        result["currency"] = "CNY"
    if "status" in result:
        result["status_label"] = status_label(result["status"])
    return result


def money(amount) -> str:
    return f"¥{float(amount):.2f}（人民币）"
