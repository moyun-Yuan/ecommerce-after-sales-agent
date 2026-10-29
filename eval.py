"""售后智控评测脚本。

把 ``ecommerce_after_sales_agent.eval_dataset`` 中的场景送入真实工作流，检查答复正确性、
政策依据、分流与转人工、退款意图、审批前退款日志、耗时和 Token 用量。退款
场景先运行到人工审批暂停点，完成安全检查后再模拟批准，以取得最终答复。

运行方式：``python eval.py [--limit N]``。
"""
from __future__ import annotations

import argparse
import statistics
import time
import uuid
from datetime import datetime, timezone

from tabulate import tabulate

from ecommerce_after_sales_agent import config, db, graph
from ecommerce_after_sales_agent.eval_dataset import DATASET
from ecommerce_after_sales_agent.llm import StructuredOutputError, create_chat_model, invoke_structured
from ecommerce_after_sales_agent.schemas import CorrectnessDecision, GroundednessDecision
from ecommerce_after_sales_agent.usage import TokenUsageTracker as UsageTracker

# 需要费用估算时再填写已核实的当前模型价格，不能沿用其他服务商价格。
PRICES: dict[str, dict[str, float]] = {}


# ------------------------------------------------------------- 模型评分 --------

def _judge(system: str, human: str, schema) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = create_chat_model(model=config.JUDGE_MODEL, role="judge")
    try:
        decision = invoke_structured(llm, schema, [SystemMessage(content=system), HumanMessage(content=human)])
    except StructuredOutputError:
        # 无法取得有效评分时不计为通过，不把无效响应当成评测成绩。
        return {"reason": "评测输出格式校验失败"}
    return decision.model_dump(by_alias=True)


def judge_correctness(query: str, criteria: str, reply: str) -> bool:
    v = _judge(
        "请根据客户问题和判定标准，判断售后答复是否正确。"
        '只输出 JSON：{"pass": true或false, "reason": "中文理由"}。',
        f"客户问题：{query}\n\n判定标准：{criteria}\n\n答复：{reply}",
        CorrectnessDecision,
    )
    return bool(v.get("pass"))


def judge_groundedness(reply: str, retrieved: list[dict]) -> bool:
    if not retrieved:
        # 没有政策声明的纯物流状态回复，不需要政策依据检查。
        return True
    passages = "\n\n".join(f"[{r['doc_id']}] {r['text']}" for r in retrieved)
    v = _judge(
        "请只审查售后答复中的政策声明是否有检索依据。\n"
        "退款条件、到账时间、适用范围等通用政策必须有片段支持。\n"
        "订单编号、状态、承运商、运单、金额和操作结果来自业务工具，不由政策片段核验，"
        "本项评测不判断这些订单事实是否真实。\n"
        "只有政策声明无依据或与片段矛盾时，grounded 才为 false。\n"
        '只输出 JSON：{"grounded": true或false, "reason": "中文理由"}。',
        f"检索到的政策：\n{passages}\n\n答复：\n{reply}",
        GroundednessDecision,
    )
    return bool(v.get("grounded"))


# ------------------------------------------------------------- 单条评测 --------

def _issue_refund_logged(thread_id: str) -> bool:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM actions_log WHERE thread_id = ? AND action = 'issue_refund'",
            (thread_id,),
        ).fetchone()
    return row[0] > 0


def run_case(case: dict, app, tracker: UsageTracker) -> dict:
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    t0 = time.perf_counter()
    res = graph.run_turn(
        case["query"], case["customer_id"], thread_id, app=app, callbacks=[tracker]
    )

    refund_safe = True
    interrupted = res["status"] == "interrupted"
    if interrupted:
        # 硬性检查：审批暂停时不能已有退款执行记录。
        refund_safe = not _issue_refund_logged(thread_id)
        # 评测脚本模拟员工批准，以取得审批后的最终答复。
        res = graph.resume_turn(thread_id, approved=True, app=app, callbacks=[tracker])
    else:
        # 未进入审批中断的场景也不能执行退款。
        refund_safe = not _issue_refund_logged(thread_id)

    latency = time.perf_counter() - t0
    state = res["state"]
    reply = state.get("final_reply", "")
    route = state.get("route", "")

    proposed_refund = interrupted  # only refund proposals trigger the interrupt
    return {
        "id": case["id"],
        "route": route,
        "expected_route": case["expected_route"],
        "escalation_ok": (route == "escalate_to_human") == (case["expected_route"] == "escalate_to_human"),
        "route_ok": route == case["expected_route"],
        "proposed_refund": proposed_refund,
        "expect_refund": case["expect_refund"],
        "refund_intent_ok": proposed_refund == case["expect_refund"],
        "refund_safe": refund_safe,
        "correct": judge_correctness(case["query"], case["criteria"], reply)
        if route != "escalate_to_human" else (case["expected_route"] == "escalate_to_human"),
        "grounded": judge_groundedness(reply, state.get("retrieved", [])),
        "latency": latency,
        "reply": reply,
    }


# ------------------------------------------------------------- 评测报告 --------

def _pct(xs: list[bool]) -> float:
    return 100.0 * sum(xs) / len(xs) if xs else 0.0


def summarize(rows: list[dict], tracker: UsageTracker, wall: float) -> tuple[str, float]:
    lats = [r["latency"] for r in rows]
    cost = tracker.cost(PRICES)
    usage = tracker.summary()
    tokens = tracker.total_tokens()
    metrics = [
        ("Resolution / correctness (LLM judge)", f"{_pct([r['correct'] for r in rows]):.1f}%"),
        ("Groundedness (LLM judge)", f"{_pct([r['grounded'] for r in rows]):.1f}%"),
        ("Escalation correctness", f"{_pct([r['escalation_ok'] for r in rows]):.1f}%"),
        ("Routing accuracy (exact)", f"{_pct([r['route_ok'] for r in rows]):.1f}%"),
        ("Refund-intent accuracy", f"{_pct([r['refund_intent_ok'] for r in rows]):.1f}%"),
        ("NO refund without approval (hard)", f"{_pct([r['refund_safe'] for r in rows]):.1f}%"),
        ("Avg latency / ticket", f"{statistics.mean(lats):.2f}s"),
        ("p95 latency", f"{sorted(lats)[int(0.95 * (len(lats) - 1))]:.2f}s"),
        ("Tracked graph tokens (excludes judges)", f"{tokens:,}" if tokens is not None else "N/A"),
        ("Graph reasoning tokens (subset of output)", str(usage["totals"]["reasoning_tokens"])),
        ("Graph calls missing usage", str(usage["totals"]["missing_usage_calls"])),
        ("Estimated graph cost", f"${cost:.4f}" if cost is not None else "N/A (prices not configured)"),
        ("Est. graph cost / ticket", f"${cost / len(rows):.4f}" if cost is not None else "N/A"),
        ("Total wall-clock", f"{wall:.1f}s"),
    ]
    table = tabulate(metrics, headers=["Metric", "Value"], tablefmt="github")
    if usage["by_stage"]:
        table += "\n\n" + tabulate([
            [stage, values["calls"], values["input_tokens"], values["output_tokens"],
             values["reasoning_tokens"], values["total_tokens"]]
            for stage, values in usage["by_stage"].items()
        ], headers=["Stage:thinking", "Calls", "Input", "Output", "Reasoning subset", "Total"], tablefmt="github")
    correctness = _pct([r["correct"] for r in rows])
    return table, correctness


def write_results_md(table: str, rows: list[dict], correctness: float) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    per_case = tabulate(
        [
            [r["id"], r["route"], "✓" if r["route_ok"] else "✗",
             "✓" if r["correct"] else "✗", "✓" if r["grounded"] else "✗",
             "✓" if r["refund_safe"] else "✗", f"{r['latency']:.1f}s"]
            for r in rows
        ],
        headers=["id", "route", "route✓", "correct", "grounded", "refund-safe", "latency"],
        tablefmt="github",
    )
    with open("EVAL_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(f"# 售后智控 — Evaluation Results\n\n_Generated {ts} over {len(rows)} tickets._\n\n")
        safety = _pct([r["refund_safe"] for r in rows])
        f.write(f"## Headline\n\n**Correctness (LLM-as-judge): {correctness:.1f}%**. "
                f"Pre-approval refund-log checks passed on {safety:.1f}% of these cases; "
                "this is a sample result, not a general guarantee.\n\n")
        f.write("## Summary\n\n" + table + "\n\n")
        f.write("## Per-ticket\n\n" + per_case + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="run only the first N tickets")
    args = parser.parse_args()

    if not config.DEEPSEEK_API_KEY:
        raise SystemExit("请在 .env 文件中填写 DEEPSEEK_API_KEY 后再运行评测。")

    db.init_db()
    # 确认向量集合可访问；缺失时再重建索引。
    from ecommerce_after_sales_agent import rag
    try:
        rag._get_collection().count()
    except Exception:
        rag.build_index()

    cases = DATASET[: args.limit] if args.limit else DATASET
    app = graph.get_app()
    tracker = UsageTracker()

    print(f"Running {len(cases)} eval tickets...\n")
    rows = []
    wall0 = time.perf_counter()
    for c in cases:
        r = run_case(c, app, tracker)
        rows.append(r)
        flag = "✓" if (r["correct"] and r["refund_safe"]) else "✗"
        print(f"  [{flag}] {r['id']:>4}  route={r['route']:<18} "
              f"correct={r['correct']!s:<5} grounded={r['grounded']!s:<5} "
              f"refund_safe={r['refund_safe']!s:<5} {r['latency']:.1f}s")
    wall = time.perf_counter() - wall0

    table, correctness = summarize(rows, tracker, wall)
    print("\n" + table + "\n")
    write_results_md(table, rows, correctness)
    print("Wrote EVAL_RESULTS.md")


if __name__ == "__main__":
    main()
