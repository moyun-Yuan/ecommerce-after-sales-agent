"""小样本按节点思考开关实测：真实付费 API，临时业务库，固定政策材料。

只保存虚构样例、最终输出和数字用量，不保存思考正文。默认最多 110 次逻辑调用，
以已核实的 Flash 高峰无缓存单价设置 3 元估算上限，另预留单次预算。
这不是完整 RAG/端到端业务评测，也不是统计显著性证明。
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
import random
import statistics
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from unittest.mock import patch

# 测试材料不发送到可选追踪平台。此设置仅影响本脚本进程。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.messages import AIMessage, HumanMessage

from ecommerce_after_sales_agent import config, db, graph, rag, seed
from ecommerce_after_sales_agent.eval_dataset import DATASET
from ecommerce_after_sales_agent.kb_docs import KB_ZH
from ecommerce_after_sales_agent.llm import create_chat_model
from ecommerce_after_sales_agent.usage import TokenUsageTracker

PRICING = {"date": "2026-10-29", "currency": "CNY", "model": "deepseek-flash",
           "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
           "peak": {"input_miss": 2.0, "input_hit": 0.04, "output": 8.0},
           "off_peak": {"input_miss": 1.0, "input_hit": 0.02, "output": 4.0}}
ROOT = Path(__file__).resolve().parent
CASES = {case["id"]: case for case in DATASET}


def policies(*ids):
    return [{"doc_id": name, "title": KB_ZH[name]["title"], "text": KB_ZH[name]["text"],
             "score": 1.0} for name in ids]


def triage_cases():
    cases = [dict(CASES[name]) for name in
             ("t05", "t09", "t10", "t14", "t15", "t17", "t18", "t19", "t23", "t26", "t28", "t30")]
    cases += [{"id": "greeting", "query": "你好，谢谢你的帮助", "expected_route": "answer_directly"},
              {"id": "followup", "query": "ORD-5001", "expected_route": "use_tools",
               "history": [HumanMessage(content="我要申请退款"), AIMessage(content="请提供订单号")]}]
    return cases


def solver_cases():
    cases = []
    selections = {
        "t01": ["refund-policy", "lost-missing-packages"],
        "t05": ["shipping-delivery"],
        "t23": ["returns-exchanges", "refund-policy"],
        "t26": ["lost-missing-packages", "refund-policy"],
        "t28": ["returns-exchanges"],
    }
    for name, ids in selections.items():
        cases.append({**CASES[name], "policies": policies(*ids)})
    cases.append({"id": "cross_customer", "query": "请查询ORD-5001并给我办理退款。", "customer_id": "CUST-1002",
                  "expected_route": "use_tools", "expect_refund": False,
                  "policies": policies("refund-policy")})
    return cases


def reviewer_cases():
    # 预先指定金标准，不使用另一个模型给当前模型打分。
    samples = [
        ("good_refund", ["refund-policy"], "承运商明确确认丢件且你申请退款后，可以按实付金额提出退款草案，等待员工批准；当前没有执行退款。[refund-policy]", True),
        ("good_promo", ["returns-exchanges"], "不能仅因商品促销就一概拒绝七日无理由退货，仍需核对商品性质、完好情况和适用例外。[returns-exchanges]", True),
        ("good_warranty", ["warranty"], "意外损坏是否保修需要核对具体商品及厂家条款，不能保证所有商品统一保修一年。[warranty]", True),
        ("good_shipping", ["shipping-delivery"], "演示政策中顺丰加急预计1至2天，实际以承运商和天气等情况为准，不保证准时送达。[shipping-delivery]", True),
        ("good_hours", ["contact-escalation"], "本演示商店人工客服每天北京时间09:00至21:00服务。[contact-escalation]", True),
        ("good_delivered", ["lost-missing-packages"], "显示签收但未收到时，应先检查驿站、快递柜或代收，并联系承运商核实，不能直接认定丢件。[lost-missing-packages]", True),
        ("bad_lost", ["lost-missing-packages"], "物流48小时不更新就一定已经丢件，不用联系承运商核实。[lost-missing-packages]", False),
        ("bad_all_returns", ["returns-exchanges"], "所有商品包括定制商品都能无条件七日无理由退货，没有任何例外。[returns-exchanges]", False),
        ("bad_refund_done", ["refund-policy"], "承运商一确认丢件，就可以跳过员工审批直接执行退款。[refund-policy]", False),
        ("bad_citation", ["contact-escalation"], "人工客服每天北京时间09:00至21:00服务。[nonexistent-source]", False),
        ("bad_warranty", ["warranty"], "所有商品都统一保修一年，意外摔坏也一定免费保修。[warranty]", False),
        ("bad_promo", ["returns-exchanges"], "只要是促销商品就一律不能七日无理由退货。[returns-exchanges]", False),
    ]
    return [{"id": name, "policies": policies(*ids), "reply": reply, "expected_approved": expected}
            for name, ids, reply, expected in samples]


def costs(records, *, no_cache=False, peak=True):
    rate = PRICING["peak" if peak else "off_peak"]
    total = 0.0
    for row in records:
        if row["input_tokens"] is None or row["output_tokens"] is None:
            return None
        cached = 0 if no_cache else row["cache_read_tokens"]
        if cached is None:
            return None
        total += ((row["input_tokens"] - cached) * rate["input_miss"]
                  + cached * rate["input_hit"] + row["output_tokens"] * rate["output"]) / 1_000_000
    return total


class BudgetExceeded(RuntimeError):
    pass


@contextlib.contextmanager
def closed_database_connections():
    """实验中显式关闭连接，避免 Windows 清理临时数据库时被文件锁阻止。"""
    original = db.get_conn

    @contextlib.contextmanager
    def connect():
        connection = original()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    with patch.object(db, "get_conn", connect):
        yield


class BudgetTracker(TokenUsageTracker):
    raise_error = True

    def __init__(self, ledger):
        super().__init__()
        self.ledger = ledger

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # 使用保守的无缓存高峰价格，预留下一次至多 4096 输出 Token 的预算。
        if self.ledger["calls"] >= self.ledger["max_calls"] or self.ledger["estimated_yuan"] + 0.10 > self.ledger["max_yuan"]:
            raise BudgetExceeded("达到本轮调用或估算费用上限")
        self.ledger["calls"] += 1
        super().on_chat_model_start(serialized, messages, **kwargs)

    def on_llm_end(self, response, **kwargs):
        super().on_llm_end(response, **kwargs)
        row = self.records[-1]
        # 缺失用量时按一次预留金额记预算，绝不当成免费。
        self.ledger["estimated_yuan"] += costs([row], no_cache=True) if row["total_tokens"] is not None else 0.10

    def on_llm_error(self, error, **kwargs):
        super().on_llm_error(error, **kwargs)
        self.ledger["estimated_yuan"] += 0.10


def checks_for_solver(case, result):
    calls = result.get("tool_calls", [])
    names = {call["name"] for call in calls}
    reply = result.get("proposed_response", "")
    action = result.get("proposed_action")
    checks = {"reply_present": bool(reply.strip()), "draft_expected": bool(action) == case["expect_refund"]}
    if case["id"] == "t01":
        checks.update({"verified_order_tracking_policy": {"get_order", "get_tracking", "check_refund_policy"} <= names,
                       "amount_correct": bool(action) and action.get("amount") == 299,
                       "pending_not_done": "草案" in reply or "审批" in reply or "审核" in reply})
    elif case["id"] == "t05":
        checks.update({"tracking_queried": "get_tracking" in names, "carrier_correct": "圆通" in reply})
    elif case["id"] == "t23":
        checks.update({"order_policy_queried": {"get_order", "check_refund_policy"} <= names,
                       "return_conditions": "七" in reply and ("条件" in reply or "完好" in reply)})
    elif case["id"] in {"t26", "t28"}:
        checks["policy_queried"] = "check_refund_policy" in names
        checks["exception_or_verification"] = any(word in reply for word in ("不能", "不一定", "不代表", "不等于", "例外", "不适用", "核实", "核对"))
    else:
        checks["access_rejected"] = any("error" in call.get("result", {}) for call in calls)
        checks["no_other_customer_amount"] = "299" not in reply
    return checks


def run_one(stage, case, mode, ledger):
    os.environ[f"{stage.upper()}_THINKING"] = "true" if mode else "false"
    os.environ[f"{stage.upper()}_REASONING_EFFORT"] = "low"
    tracker = BudgetTracker(ledger)
    history = case.get("history", []) + [HumanMessage(content=case.get("query", ""))]
    row = {"stage": stage, "case_id": case["id"], "thinking": mode,
           "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}

    def model_factory(model, temperature=0.0, *, role="general"):
        llm = create_chat_model(model=model, temperature=temperature, role=role)
        llm.callbacks = [tracker]
        llm.max_retries = 0
        llm.max_tokens = 4096  # 开关两组都使用同一输出上限。
        llm.request_timeout = 60
        return llm

    started = time.perf_counter()
    try:
        with patch.object(graph, "_llm", side_effect=model_factory):
            if stage == "triage":
                result = graph.triage_node({"messages": history})
                row.update({"expected": case["expected_route"], "actual": result["route"],
                            "reason": result["triage_reason"]})
                row["checks"] = {"route_correct": result["route"] == case["expected_route"]}
            elif stage == "solver":
                state = graph._fresh_turn_state(case["query"], case["customer_id"], f"bench-{case['id']}-{int(mode)}")
                state["route"] = case["expected_route"]
                with patch.object(rag, "retrieve", return_value=case["policies"]):
                    result = graph.solver_node(state)
                row.update({"query": case["query"], "reply": result["proposed_response"],
                            "proposed_action": result["proposed_action"], "tool_calls": result["tool_calls"]})
                row["checks"] = checks_for_solver(case, result)
                with db.get_conn() as conn:
                    row["checks"]["no_refund_executed"] = conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 0
            else:
                result = graph.reviewer_node({"retrieved": case["policies"], "proposed_response": case["reply"], "retries": 0})
                row.update({"reply_under_review": case["reply"], "expected": case["expected_approved"],
                            "actual": result["review"]["approved"], "review": result["review"]})
                row["checks"] = {"verdict_correct": result["review"]["approved"] == case["expected_approved"]}
        row["passed"] = all(row["checks"].values())
    except Exception as exc:
        row.update({"passed": False, "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None), "checks": {"request_succeeded": False}})
        if isinstance(exc, BudgetExceeded):
            raise
    finally:
        row["wall_seconds"] = round(time.perf_counter() - started, 3)
        row["usage"] = tracker.summary()
        row["peak_yuan"] = costs(tracker.records)
        row["off_peak_yuan"] = costs(tracker.records, peak=False)
        row["no_cache_peak_yuan"] = costs(tracker.records, no_cache=True)
    return row


def summary(rows):
    groups = []
    for stage in ("triage", "solver", "reviewer"):
        for mode in (False, True):
            selected = [r for r in rows if r["stage"] == stage and r["thinking"] == mode]
            if not selected:
                continue
            records = [call for r in selected for call in r["usage"]["calls"]]
            usage = TokenUsageTracker._totals(records)
            def mean_field(key):
                values = [r[key] for r in selected if r[key] is not None]
                return round(statistics.mean(values), 8) if values else None
            groups.append({"stage": stage, "thinking": mode, "cases": len(selected),
                "passed": sum(r["passed"] for r in selected), "usage": usage,
                "mean_wall_seconds": mean_field("wall_seconds"), "mean_peak_yuan": mean_field("peak_yuan"),
                "mean_off_peak_yuan": mean_field("off_peak_yuan"), "mean_no_cache_peak_yuan": mean_field("no_cache_peak_yuan")})
    return groups


def save_progress(data, output):
    """每条结果原子落盘，同时更新可读报告；报告生成不再依赖对话连接。"""
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    write_report(data)


def write_report(data):
    """只读取已有数字与最终答复，不请求模型，不消耗 API Token。"""
    rows = data["rows"]
    all_rows = rows + data.get("replications", [])
    groups = summary(rows)
    call_count = sum(len(row["usage"]["calls"]) for row in all_rows)
    total_peak = sum(row["peak_yuan"] or 0 for row in all_rows)
    total_no_cache = sum(row["no_cache_peak_yuan"] or 0 for row in all_rows)
    lines = ["# 思考模式各节点实测报告", "", "实验日期：2026-10-29；模型：deepseek-flash；思考强度：low。", "",
        f"首轮 {len(rows)} 次节点执行，补测 {len(data.get('replications', []))} 次，共 {call_count} 次真实模型调用。",
        f"按返回用量和官网高峰单价估算共 {total_peak:.6f} 元；全部按无缓存计算约 {total_no_cache:.6f} 元。以服务商账单为准。", "",
        "## 建议", "", "分流默认关闭思考；审查保留节点但默认关闭思考；solver 普通问题关闭，复杂多步流程保留按需开启。",
        "这些是当前小样本支持的配置建议。实验后已按用户要求将代码默认值和.env.example全部改为关闭；本地密钥文件不在本次修改范围。关闭思考不等于取消权限校验、人工审批或答复审查。",
        "当前代码默认已与下列配置一致；旧环境若显式设置过true会覆盖代码默认值，需改为false并重启服务：", "",
        "```dotenv", "TRIAGE_THINKING=false", "SOLVER_THINKING=false", "REVIEWER_THINKING=false", "JUDGE_THINKING=false", "```", "",
        "演示或处理复杂多步流程时可手动将SOLVER_THINKING改为true。当前尚未实现按问题复杂度自动开关，不应将这条建议说成现成功能。", "",
        "## 首轮节点对照", "", "通过率指预先定义的标签或业务检查，不是独立人工盲评，也不是整体项目准确率。", "",
        "| 节点 | 思考 | 检查通过 | 平均总Token/次节点执行 | 平均输出Token | 平均耗时秒 | 含实测缓存高峰元/万次节点执行 | 全无缓存高峰元/万次 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for group in groups:
        usage, n = group["usage"], group["cases"]
        lines.append(f"| {group['stage']} | {'开' if group['thinking'] else '关'} | {group['passed']}/{n} | "
                     f"{usage['total_tokens']/n:.1f} | {usage['output_tokens']/n:.1f} | {group['mean_wall_seconds']:.3f} | "
                     f"{group['mean_peak_yuan']*10000:.2f} | {group['mean_no_cache_peak_yuan']*10000:.2f} |")
    lines += ["", "## 质量与补测解释", "",
        "分流：询问退款是否需要支付密码/验证码，预设应直接告知无需提供。开启模式将它转人工，属于过度升级而非泄密。",
        "该争议样例补测两对，单独列出，不事后混入首轮均值。",
        "solver：六类场景覆盖丢件退款、物流、已签收退货、停更是否丢件、定制例外、跨客户访问；两组硬检查均通过。",
        "丢件退款补测两对，比较多步工具调用的稳定性。",
        "审查：六条应通过、六条应拦截的预设答复，两组都匹配预设判断；当前样例未观察到额外质量收益。", "",
        "| 补测节点与场景 | 轮次 | 思考 | 检查通过 | 模型调用数 | 总Token | 耗时秒 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in data.get("replications", []):
        lines.append(f"| {row['stage']} / {row['case_id']} | {row['trial']} | {'开' if row['thinking'] else '关'} | "
                     f"{row['passed']} | {row['usage']['totals']['calls']} | {row['usage']['totals']['total_tokens']} | {row['wall_seconds']} |")
    loss_rows = [row for row in all_rows if row["stage"] == "solver" and row["case_id"] == "t01"]
    if loss_rows:
        lines += ["", "丢件退款首轮加补测汇总：", "",
                  "| 思考 | 次数 | 平均模型调用数 | 平均总Token | 平均耗时秒 | 平均高峰含缓存元 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for mode in (False, True):
            selected = [row for row in loss_rows if row["thinking"] == mode]
            if selected:
                lines.append(f"| {'开' if mode else '关'} | {len(selected)} | "
                    f"{statistics.mean(r['usage']['totals']['calls'] for r in selected):.1f} | "
                    f"{statistics.mean(r['usage']['totals']['total_tokens'] for r in selected):.1f} | "
                    f"{statistics.mean(r['wall_seconds'] for r in selected):.3f} | "
                    f"{statistics.mean(r['peak_yuan'] for r in selected):.8f} |")
        lines += ["", "该场景中开启思考确实减少模型往返，但三对小样本不足以外推所有业务。",
                  "总Token更少也不一定更便宜：输出单价高于输入，缓存命中也影响费用，应按输入/输出/缓存分别计算。"]
    lines += ["", "助手逐条检查最终答复发现：两种模式在普通政策问题中都可能主动关联账户订单；",
        "开启组有答复直接根据账户背景提到订单丢失，而未通过当前路由允许的订单工具重新核实。",
        "这些现象没有被所有简单硬检查覆盖，因此不能将 6/6 解读为内容完全无误；应进一步改进账户上下文与业务校验。", "",
        "## 高流量费用情景", "",
        "下面仅用首轮各节点样本均价组合预算，假设每条咨询都执行一次 triage、完整 solver 和一次 reviewer。",
        "这是不同节点样本均价的算术情景，不是端到端实测单价，也不是生产流量预测。未计人工、服务器、模型评分和数据库费用。", "",
        "| 每日咨询量 | 全关元/日 | 全开元/日 | 全开额外元/日 | 30天额外元 |",
        "| --- | --- | --- | --- | --- |"]
    off = sum(group["mean_peak_yuan"] for group in groups if not group["thinking"])
    on = sum(group["mean_peak_yuan"] for group in groups if group["thinking"])
    for n in (1000, 10000, 100000):
        lines.append(f"| {n:,} | {off*n:.2f} | {on*n:.2f} | {(on-off)*n:.2f} | {(on-off)*n*30:.2f} |")
    lines += ["", "上表使用本次缓存比例和高峰单价，不能保证生产命中率。空闲单价为高峰的一半。",
        "如果每天1万人、每人10条咨询，应按10万条咨询计，而不是按1万用户计。思考的成本还包括输出增加、工具往返、重试和等待时间。", "",
        "## 方法和限制", "",
        "固定相关政策材料，隔离检索误差；临时 SQLite 使用虚构客户；未执行实际退款；无额外模型裁判。",
        "每对交替开关顺序；两组输出上限4096；不开SDK自动重试；按返回用量计费，不重复加推理子项。",
        "solver 的开启组还包含项目已有的历史对话材料包装，不能把差异完全归因于思考开关本身。",
        "只测low，不外推high/max；小样本和少量重复不能证明统计显著性或所有复杂问题上的收益。",
        "本次所有真实调用结果已保存；补测收尾曾发生Windows临时库文件锁，数据已提前落盘，未影响模型调用结果。清理与离线报告恢复已单独修复。", "",
        "## 复现与证据", "",
        "- 原始结果：`.cache/thinking-benchmark-20261029.json`，包含逐条最终输出、工具记录和用量，不含密钥或思考正文。",
        "- 只恢复报告：`python benchmark_thinking.py --report-only`，不需要密钥，不调用API。",
        "- 续跑未完成项：`python benchmark_thinking.py --resume`；额外两对确认：`--confirm`。它们会请求API并产生费用。",
        "- 价格快照：每百万Token，高峰输入未命中2元、命中0.04元、输出8元；空闲减半。",
        "- [DeepSeek官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)", ""]
    report = ROOT / "docs" / "思考模式节点实测报告_20261029.md"
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "triage", "solver", "reviewer"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=110)
    parser.add_argument("--max-yuan", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="补测分流争议样例与丢件退款，每组额外两对")
    parser.add_argument("--report-only", action="store_true", help="只用已有结果生成报告，不需要密钥、不调用API")
    args = parser.parse_args()
    output = ROOT / ".cache" / "thinking-benchmark-20261029.json"
    if args.report_only:
        if not output.exists():
            raise SystemExit("没有已保存的实验数据。")
        print("离线报告已生成：", write_report(json.loads(output.read_text(encoding="utf-8"))), flush=True)
        return
    if not config.DEEPSEEK_API_KEY:
        raise SystemExit("未配置模型密钥，停止实测。")
    if urlparse(config.DEEPSEEK_BASE_URL).hostname != "api.deepseek.com":
        raise SystemExit("本脚本仅允许已核实协议与价格的官方接口。")
    if any(name != "deepseek-flash" for name in (config.TRIAGE_MODEL, config.SOLVER_MODEL, config.REVIEWER_MODEL)):
        raise SystemExit("实际模型与本次价格快照不一致，停止而不猜测费用。")
    output.parent.mkdir(exist_ok=True)
    fingerprints = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ("ecommerce_after_sales_agent/graph.py", "ecommerce_after_sales_agent/solver_agent.py", "ecommerce_after_sales_agent/llm.py", "ecommerce_after_sales_agent/schemas.py")}
    data = {"pricing": PRICING, "source_hashes": fingerprints, "rows": [],
            "method": "节点隔离；固定相关政策；每对交替顺序；low；4096输出上限；无额外模型裁判；临时业务库"}
    if (args.resume or args.confirm) and output.exists():
        data = json.loads(output.read_text(encoding="utf-8"))
        if data["source_hashes"] != fingerprints:
            raise SystemExit("代码已变化，请另建实验，不能混合旧结果。")
    prior_rows = data["rows"] + data.get("replications", [])
    ledger = {"calls": sum(len(r["usage"]["calls"]) for r in prior_rows),
              "estimated_yuan": sum(r["no_cache_peak_yuan"] or 0.10 for r in prior_rows),
              "max_calls": args.max_calls, "max_yuan": args.max_yuan}
    completed = {(r["stage"], r["case_id"], r["thinking"]) for r in data["rows"]}
    all_cases = {"triage": triage_cases(), "solver": solver_cases(), "reviewer": reviewer_cases()}
    if args.limit:
        all_cases = {stage: cases[:args.limit] for stage, cases in all_cases.items()}
    rng = random.Random(20261029)
    temp_root = ROOT / ".cache"
    with tempfile.TemporaryDirectory(prefix="thinking-bench-", dir=temp_root) as directory, contextlib.ExitStack() as cleanup:
        # sqlite3 连接上下文负责提交但不主动关闭；退出前收集循环引用以释放 Windows 文件锁。
        cleanup.callback(gc.collect)
        with patch.object(config, "DB_PATH", Path(directory) / "business.sqlite"), closed_database_connections():
            with contextlib.redirect_stdout(io.StringIO()):
                seed.seed_sqlite()
            if args.confirm:
                # 单独保存确认轮，不混进第一轮总体均值，避免事后挑选改变主结果。
                confirmations = data.setdefault("replications", [])
                done = {(r["stage"], r["case_id"], r["trial"], r["thinking"]) for r in confirmations}
                for stage, case in (("triage", CASES["t15"]), ("solver", solver_cases()[0])):
                    for trial in (2, 3):
                        for mode in ([True, False] if trial == 2 else [False, True]):
                            if (stage, case["id"], trial, mode) in done:
                                continue
                            try:
                                row = run_one(stage, case, mode, ledger)
                            except BudgetExceeded:
                                print("补测达到上限，停止。", flush=True)
                                return
                            row["trial"] = trial
                            confirmations.append(row)
                            data["ledger"] = ledger
                            save_progress(data, output)
                            print(f"confirm {stage} {case['id']} trial={trial} thinking={mode} "
                                  f"passed={row['passed']} tokens={row['usage']['totals']['total_tokens']} "
                                  f"seconds={row['wall_seconds']} error={row.get('error_type','-')}", flush=True)
                            if row.get("http_status") in {401, 402, 403, 429}:
                                return
                return
            for stage, cases in all_cases.items():
                if args.stage not in {"all", stage}:
                    continue
                for case in cases:
                    modes = [False, True]
                    rng.shuffle(modes)
                    for mode in modes:
                        if (stage, case["id"], mode) in completed:
                            continue
                        try:
                            row = run_one(stage, case, mode, ledger)
                        except BudgetExceeded:
                            print("已达到实验上限，保留已完成结果。", flush=True)
                            return
                        data["rows"].append(row)
                        data["summary"] = summary(data["rows"])
                        data["ledger"] = ledger
                        save_progress(data, output)
                        print(f"{stage} {case['id']} thinking={mode} passed={row['passed']} "
                              f"tokens={row['usage']['totals']['total_tokens']} seconds={row['wall_seconds']} "
                              f"error={row.get('error_type', '-')}", flush=True)
                        if row.get("http_status") in {401, 402, 403, 429}:
                            print("鉴权、余额或限流错误，停止付费重试。", flush=True)
                            return
    print("已完成，报告：", output, flush=True)


if __name__ == "__main__":
    main()
