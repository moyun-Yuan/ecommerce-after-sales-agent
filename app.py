"""售后智控 — Streamlit demo UI (two roles, async approval).

This single app serves TWO different humans, switched via the sidebar role toggle —
because in reality a customer never approves their own refund:

  👤 Customer  — chats with the agent. When the AI proposes a sensitive action
                 (a refund), the customer is told a specialist will review it and
                 the conversation ENDS for them; they are never blocked. Later,
                 once staff decide, the outcome shows up as a follow-up message.

  🧑‍💼 Staff    — sees the pending-approval QUEUE (the `approvals` table). Each item
                 shows what the AI proposed, with full context, and Approve/Reject
                 resumes that durable thread out-of-band via Command(resume=...).

This works because LangGraph's interrupt() pauses the run to the SQLite checkpoint
(keyed by thread_id) and returns control right away, so the staff decision can
happen seconds or hours later, from this other view or process.

Run:  streamlit run app.py
"""
from __future__ import annotations

import uuid

import streamlit as st

from ecommerce_after_sales_agent import config, db, graph
from ecommerce_after_sales_agent.i18n import (
    ROUTE_NAMES, TOOL_NAMES, customer_label, localize_order, money, status_label,
)
from ecommerce_after_sales_agent.kb_docs import KB_ZH

st.set_page_config(page_title="电商售后智能助手", page_icon="🛟", layout="wide")


# ------------------------------------------------------------- 缓存与数据 -----

@st.cache_resource
def get_app(profile: str):
    """在 Streamlit 重运行之间复用同一个工作流和 SQLite 检查点。"""
    db.init_db()
    return graph.get_app()


@st.cache_data
def list_customers(profile: str) -> list[tuple[str, str]]:
    with db.get_conn() as conn:
        return [(r["id"], customer_label(r["id"], r["name"]))
                for r in conn.execute("SELECT id, name FROM customers ORDER BY id")]


@st.cache_data
def customer_name(customer_id: str, profile: str) -> str:
    with db.get_conn() as conn:
        row = conn.execute("SELECT name FROM customers WHERE id = ?", (customer_id,)).fetchone()
    return customer_label(customer_id, row["name"]) if row else customer_id


@st.cache_data
def orders_for(customer_id: str, profile: str) -> list[dict]:
    with db.get_conn() as conn:
        return [localize_order(dict(r)) for r in conn.execute(
            "SELECT id, item, amount, status FROM orders WHERE customer_id = ? ORDER BY id",
            (customer_id,))]


# 按订单状态提供可直接复制的演示问题。
_SUGGEST = {
    "lost": "我的订单 {oid} 一直没收到，想申请退款。",
    "in_transit": "请帮我查一下订单 {oid} 的物流。",
    "delivered": "订单 {oid} 已收到，我想咨询退货退款政策。",
    "processing": "订单 {oid} 什么时候发货？",
}


def new_thread() -> None:
    """为客户视图创建新的会话。"""
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.session_state.awaiting = []      # 等待员工审批的暂停会话
    st.session_state.last_state = {}
    st.session_state.last_usage = {}


def actions_for_thread(thread_id: str) -> list[dict]:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT action, payload, created_at FROM actions_log WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------- 页面状态初始化 --

if st.session_state.get("data_profile") != config.DATA_PROFILE:
    new_thread()
    st.session_state.data_profile = config.DATA_PROFILE
if st.session_state.get("role") not in ("customer", "staff"):
    st.session_state.role = "staff" if st.session_state.get("role") == "🧑‍💼 Staff" else "customer"

app = get_app(config.DATA_PROFILE)


# ============================================================= 客户视图 ========

def check_for_updates() -> int:
    """轮询已处理的退款申请，并把最终结果追加到客户消息中。

    对等待队列中的会话检查审批状态；处理完成后读取工作流保存的最终答复，
    作为助手消息展示。返回本次展示的更新数量。
    """
    if not st.session_state.awaiting:
        return 0
    pending_threads = {p["thread_id"] for p in db.get_pending_approvals()}
    surfaced = 0
    still_waiting = []
    for item in st.session_state.awaiting:
        if item["thread_id"] in pending_threads:
            still_waiting.append(item)
            continue
        resp = db.latest_response_for_thread(item["thread_id"])
        if resp:
            st.session_state.messages.append({"role": "assistant", "content": "📬 " + resp["reply"]})
            surfaced += 1
        else:
            still_waiting.append(item)  # decided but not yet persisted; check again
    st.session_state.awaiting = still_waiting
    return surfaced


def render_customer(customer_id: str) -> None:
    st.title("💬 电商售后智能助手")
    st.caption("电商售后演示 · 人民币结算 · 北京时间 · 退款方案由员工审核")
    if not config.DEEPSEEK_API_KEY:
        st.error("请在项目 .env 文件中填写 DEEPSEEK_API_KEY，然后重启应用。")
        st.stop()

    # 检查上次交互之后是否有员工完成审批。
    check_for_updates()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if st.session_state.awaiting:
        n = len(st.session_state.awaiting)
        st.info(f"⏳ 有 {n} 个申请正在等待员工处理，你可以继续咨询，处理结果将在这里更新。")
        if st.button("🔄 查看处理结果"):
            st.rerun()

    if prompt := st.chat_input("请输入问题，例如：我的订单 ORD-5001 没收到，想申请退款。"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("正在分析问题、查询信息并检查答复…"):
            res = graph.run_turn(prompt, customer_id, st.session_state.thread_id, app=app)
        st.session_state.last_state = res.get("state", {})
        st.session_state.last_usage = res.get("usage", {})

        if res["status"] == "interrupted":
            # 敏感操作转交员工；客户侧只展示已提交提示，不提供审批按钮。暂停会话
            # 放入等待队列，同时切换到新会话，使客户仍能继续咨询。
            info = res["interrupt"]
            ref = f"A-{info.get('approval_id')}"
            ack = (
                f"✅ 已核查相关信息，你的申请已提交员工审核（编号 **{ref}**）。"
                "处理完成后将在这里通知你，期间可以继续咨询。"
            )
            st.session_state.messages.append({"role": "assistant", "content": ack})
            st.session_state.awaiting.append({
                "thread_id": st.session_state.thread_id,
                "ref": ref,
                "order_id": info.get("order_id"),
                "amount": info.get("amount"),
            })
            st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"  # rotate
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": res["state"].get("final_reply", "暂未生成回复，请稍后重试。")})
        st.rerun()


# ================================================================ 员工视图 =====

def render_staff() -> None:
    st.title("🧑‍💼 员工工作台 · 待审批申请")
    st.caption("查看助手提出的退款方案，批准后模拟执行，拒绝则不执行；决定后自动恢复处理流程。")

    pending = db.get_pending_approvals()
    if not pending:
        st.success("🎉 当前没有待审批申请。")
        if st.button("🔄 刷新列表"):
            st.rerun()
        return

    st.markdown(f"共有 **{len(pending)}** 个申请等待处理：")
    for p in pending:
        thread_id = p["thread_id"]
        payload = p["payload"]
        # 只读暂停会话中的草稿和来源，不推进工作流。
        state = graph.peek_state(thread_id, app=app)
        cust_id = state.get("customer_id", "?")
        proposed = state.get("proposed_response", "")
        citations = state.get("citations", [])

        with st.container(border=True):
            st.markdown(
                f"### 💸 退款 {money(payload.get('amount', 0))} · 订单 `{payload.get('order_id')}`")
            st.markdown(f"**客户：** {customer_name(cust_id, config.DATA_PROFILE)} (`{cust_id}`) · "
                        f"申请 **A-{p['id']}** · 会话 `{thread_id}`")
            if proposed:
                with st.expander("🤖 助手拟定的客户答复", expanded=True):
                    st.markdown(proposed)
            if citations:
                st.caption("政策依据：" + "、".join(KB_ZH.get(c, {}).get("title", c) for c in citations))

            c1, c2, _ = st.columns([1, 1, 3])
            if c1.button("✅ 批准", key=f"ap-{p['id']}", type="primary"):
                with st.spinner("正在模拟退款并生成最终答复…"):
                    graph.resume_turn(thread_id, approved=True, decided_by="staff", app=app)
                st.toast(f"申请 A-{p['id']} 已批准，模拟退款已记录。", icon="✅")
                st.rerun()
            if c2.button("❌ 拒绝", key=f"rj-{p['id']}"):
                with st.spinner("正在记录拒绝决定并生成答复…"):
                    graph.resume_turn(thread_id, approved=False, decided_by="staff", app=app)
                st.toast(f"申请 A-{p['id']} 已拒绝。", icon="❌")
                st.rerun()


# ------------------------------------------------------------- 侧边栏 ---------

with st.sidebar:
    st.header("🛟 售后智能助手")
    st.caption("订单查询 · 政策问答 · 人工审批")
    st.caption("金额单位：人民币（CNY）")

    st.session_state.role = st.radio(
        "查看身份", ["customer", "staff"],
        format_func=lambda role: {"customer": "👤 客户", "staff": "🧑‍💼 员工"}[role],
        index=0 if st.session_state.role == "customer" else 1,
        help="客户提交问题，员工审核需要人工确认的操作。",
    )
    is_customer = st.session_state.role == "customer"

    # 实时显示待审批数量。
    queue_depth = len(db.get_pending_approvals())
    if queue_depth:
        st.warning(f"🔔 员工工作台有 {queue_depth} 个待审批申请")

    customer_id = None
    if is_customer:
        customers = list_customers(config.DATA_PROFILE)
        labels = [f"{cid} — {name}" for cid, name in customers]
        if not customers:
            st.info("请先在终端运行 python -m ecommerce_after_sales_agent.seed 初始化演示数据。")
            st.stop()
        idx = st.selectbox("选择演示客户", range(len(customers)), format_func=lambda i: labels[i])
        new_id = customers[idx][0]
        # 切换客户时创建全新的会话。
        if st.session_state.get("customer_id") != new_id:
            st.session_state.customer_id = new_id
            new_thread()
        customer_id = new_id

        if st.button("🔄 新建对话", use_container_width=True):
            new_thread()
            st.rerun()

        # 展示当前客户的演示订单和可复制测试问题。
        st.divider()
        st.subheader("🧪 当前客户的演示订单")
        status_emoji = {"lost": "📦❌", "in_transit": "🚚", "delivered": "✅", "processing": "⏳"}
        for o in orders_for(customer_id, config.DATA_PROFILE):
            st.markdown(f"{status_emoji.get(o['status'], '•')} **{o['id']}** — {o['item']} "
                        f"· {money(o['amount'])} · {status_label(o['status'])}")
            if _SUGGEST.get(o["status"]):
                st.caption(f"试着问：“{_SUGGEST[o['status']].format(oid=o['id'])}”")

        with st.expander("📚 全部演示客户与订单"):
            for cid, name in list_customers(config.DATA_PROFILE):
                st.markdown(f"**{cid} — {name}**")
                for o in orders_for(cid, config.DATA_PROFILE):
                    st.caption(f"{o['id']} · {o['item']} · {money(o['amount'])} · {status_label(o['status'])}")

        # 展示上一轮的路由、工具调用和审查结果，便于本地排查。
        st.divider()
        st.subheader("🔎 本次执行过程")
        state = st.session_state.get("last_state", {})
        if state:
            if state.get("route"):
                st.markdown(f"**处理方式：** {ROUTE_NAMES.get(state['route'], state['route'])}")
                if state.get("triage_reason"):
                    st.caption(state["triage_reason"])
            for tc in state.get("tool_calls") or []:
                st.markdown(f"- **{TOOL_NAMES.get(tc['name'], tc['name'])}** · `{tc['name']}`")
                st.json(tc.get("args", {}), expanded=False)
            if state.get("citations"):
                st.markdown("**政策来源：** " + "、".join(KB_ZH.get(c, {}).get("title", c) for c in state["citations"]))
            review = state.get("review") or {}
            if review:
                st.markdown(f"**答复审查：** {'✅ 通过' if review.get('approved') else '❌ 未通过'}")
        usage = st.session_state.get("last_usage", {})
        if usage:
            with st.expander("本次运行的 Token 用量"):
                st.caption("仅统计最近一次运行至完成或暂停的区间；审批恢复另计。未返回的字段显示为空。")
                st.json(usage)

    st.divider()
    if config.setup_tracing():
        st.success("LangSmith 追踪已开启")
        st.markdown("[查看追踪记录 ↗](https://smith.langchain.com/)")
    else:
        st.caption("LangSmith 追踪未开启")


# ------------------------------------------------------------- 主区域 ---------

if is_customer:
    render_customer(customer_id)
else:
    render_staff()
