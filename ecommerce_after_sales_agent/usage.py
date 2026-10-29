"""仅记录用量和耗时，不记录提示词、思考正文、密钥或客户资料。"""
from __future__ import annotations

import time
from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler


class TokenUsageTracker(BaseCallbackHandler):
    """按真实响应统计；推理 Token 是输出的子项，不再加进总量。"""

    def __init__(self):
        self.records: list[dict] = []
        self._pending: dict = {}
        self._lock = Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, **kwargs):
        tags = tags or []
        stage = next((tag[6:] for tag in tags if tag.startswith("stage:")), "unknown")
        thinking = next((tag[9:] for tag in tags if tag.startswith("thinking:")), "unknown")
        with self._lock:
            self._pending[run_id] = {"stage": stage, "thinking": thinking, "started": time.perf_counter()}

    def _finish(self, run_id):
        with self._lock:
            pending = self._pending.pop(run_id, {})
        started = pending.get("started")
        return {"stage": pending.get("stage", "unknown"), "thinking": pending.get("thinking", "unknown"),
                "latency_seconds": round(time.perf_counter() - started, 4) if started is not None else None}

    def on_llm_end(self, response, *, run_id, **kwargs):
        record = self._finish(run_id)
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        raw = (response.llm_output or {}).get("token_usage") or message.response_metadata.get("token_usage") or {}
        input_tokens = usage.get("input_tokens", raw.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", raw.get("completion_tokens"))
        total = usage.get("total_tokens", raw.get("total_tokens"))
        if total is None and input_tokens is not None and output_tokens is not None:
            total = input_tokens + output_tokens
        record.update({
            "model": message.response_metadata.get("model_name") or (response.llm_output or {}).get("model_name", "unknown"),
            "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total,
            "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning",
                (raw.get("completion_tokens_details") or {}).get("reasoning_tokens")),
            "cache_read_tokens": (usage.get("input_token_details") or {}).get("cache_read", raw.get("prompt_cache_hit_tokens")),
            "failed": False,
        })
        with self._lock:
            self.records.append(record)

    def on_llm_error(self, error, *, run_id, **kwargs):
        record = self._finish(run_id)
        record.update({"model": "unknown", "input_tokens": None, "output_tokens": None,
                       "total_tokens": None, "reasoning_tokens": None, "cache_read_tokens": None, "failed": True})
        with self._lock:
            self.records.append(record)

    @staticmethod
    def _totals(records):
        result = {"calls": len(records), "failed_calls": sum(r["failed"] for r in records),
                  "missing_usage_calls": sum(r["total_tokens"] is None for r in records),
                  "reasoning_reported_calls": sum(r["reasoning_tokens"] is not None for r in records),
                  "cache_reported_calls": sum(r["cache_read_tokens"] is not None for r in records)}
        for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "cache_read_tokens"):
            known = [r[key] for r in records if r[key] is not None]
            result[key] = sum(known) if known else None
        return result

    def summary(self):
        with self._lock:
            records = [dict(r) for r in self.records]
        groups = {}
        for record in records:
            key = f"{record['stage']}:{record['thinking']}"
            groups.setdefault(key, []).append(record)
        return {"totals": self._totals(records),
                "by_stage": {key: self._totals(rows) for key, rows in groups.items()}, "calls": records}

    def total_tokens(self):
        return self.summary()["totals"]["total_tokens"]

    def cost(self, prices):
        """未配置价格或缺失用量时不估算；价格须涵盖同一模型的两种模式。"""
        records = self.summary()["calls"]
        if not records:
            return None
        total = 0.0
        for record in records:
            price = prices.get(record["model"])
            if not price or record["input_tokens"] is None or record["output_tokens"] is None:
                return None
            # 缓存计费不同而价格未配置时，不能给出看似精确的估算。
            if record["cache_read_tokens"] and "cache_read" not in price:
                return None
            cached = record["cache_read_tokens"] or 0
            total += ((record["input_tokens"] - cached) * price["in"]
                      + cached * price.get("cache_read", price["in"])
                      + record["output_tokens"] * price["out"]) / 1_000_000
        return total
