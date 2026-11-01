"""部分退款、审批金额调整和旧数据迁移测试，仅使用临时数据库和脚本模型。"""
from __future__ import annotations

import gc
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.checkpoint.memory import InMemorySaver

import api
from ecommerce_after_sales_agent import config, db, graph, seed, tools
from ecommerce_after_sales_agent.money import to_cents


LEGACY_REFUNDS = """CREATE TABLE refunds (
    id TEXT PRIMARY KEY, order_id TEXT NOT NULL UNIQUE REFERENCES orders(id),
    customer_id TEXT NOT NULL REFERENCES customers(id), amount REAL NOT NULL,
    currency TEXT NOT NULL, status TEXT NOT NULL, thread_id TEXT, created_at TEXT NOT NULL
)"""


class FakeSolver(FakeMessagesListChatModel):
    """仅模拟模型消息；工具、审批节点、状态图和数据库仍执行真实代码。"""

    def __init__(self, amount):
        super().__init__(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "create_refund_draft", "args": {"order_id": "ORD-5001", "amount": amount},
                "id": "draft-call", "type": "tool_call",
            }]),
            AIMessage(content="本次退款草案等待员工审核。"),
        ])

    def bind_tools(self, available, **kwargs):
        return self


class PartialRefundTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="partial-refund-", dir=cache)
        self.database = Path(self.temporary.name) / "business.sqlite"
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(config, "DB_PATH", self.database))
        with redirect_stdout(io.StringIO()):
            seed.seed_sqlite()
        tools.set_run_context("partial-test", "CUST-1001")

    def tearDown(self):
        self.stack.close()
        gc.collect()
        self.temporary.cleanup()

    def refund(self, amount, request_id, order="ORD-5001", customer="CUST-1001"):
        return tools.issue_refund(order, amount, "partial-test", customer, request_id=request_id)

    def totals(self):
        with closing(db.get_conn()) as conn:
            return tuple(conn.execute("SELECT COUNT(*), COALESCE(SUM(amount_cents), 0) FROM refunds").fetchone())

    def workflow(self, amounts, reviews=None):
        self.stack.enter_context(patch.object(graph, "triage_node", return_value={"route": "use_tools"}))
        self.stack.enter_context(patch.object(graph, "reviewer_node", side_effect=reviews or [
            {"review": {"approved": True}},
        ] * 10))
        self.stack.enter_context(patch.object(graph, "_llm", side_effect=[FakeSolver(x) for x in amounts]))
        return graph.build_graph(InMemorySaver())

    def start(self, app, thread):
        result = graph.run_turn("申请部分退款", "CUST-1001", thread, app=app)
        self.assertEqual(result["status"], "interrupted")
        return result["interrupt"]["approval_id"]

    def legacy_table(self, amount=299):
        with closing(db.get_conn()) as conn, conn:
            conn.execute("DROP TABLE refunds")
            conn.execute(LEGACY_REFUNDS)
            conn.execute("INSERT INTO refunds VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         ("RF-ORD-5001", "ORD-5001", "CUST-1001", amount,
                          "CNY", "refunded", "legacy-thread", db.now()))

    def test_multiple_partial_refunds_then_remaining_balance(self):
        for amount, key in [(100, "first"), (50, "second"), (149, "last")]:
            self.assertEqual(self.refund(amount, key)["status"], "refunded")
        self.assertEqual(self.totals(), (3, 29900))
        order = json.loads(tools.get_order.invoke({"order_id": "ORD-5001"}))
        self.assertEqual(order["refunded_amount"], 299)
        self.assertEqual(order["remaining_amount"], 0)
        draft = json.loads(tools.create_refund_draft.invoke({"order_id": "ORD-5001", "amount": 0.01}))
        self.assertEqual(draft["status"], "rejected")

    def test_cumulative_over_refund_is_rejected(self):
        self.refund(200, "first")
        with self.assertRaisesRegex(ValueError, "剩余"):
            self.refund(100, "over-limit")
        self.assertEqual(self.totals(), (1, 20000))

    def test_same_request_is_idempotent_and_parameters_cannot_change(self):
        first = self.refund(50, "same")
        repeated = self.refund(50, "same")
        self.assertEqual(repeated["status"], "already_refunded")
        self.assertEqual(first["refund_id"], repeated["refund_id"])
        with self.assertRaisesRegex(ValueError, "编号"):
            self.refund(60, "same")
        with self.assertRaisesRegex(ValueError, "编号"):
            self.refund(50, "same", "ORD-5002", "CUST-1002")
        self.assertEqual(self.totals(), (1, 5000))

    def test_distinct_requests_can_refund_equal_amounts(self):
        self.refund(50, "one")
        self.refund(50, "two")
        self.assertEqual(self.totals(), (2, 10000))

    def test_decimal_cents_do_not_accumulate_float_error(self):
        with closing(db.get_conn()) as conn, conn:
            conn.execute("UPDATE orders SET amount = 0.30 WHERE id = 'ORD-5001'")
        self.refund(0.10, "ten-cents")
        last = self.refund(0.20, "twenty-cents")
        self.assertEqual(last["remaining_amount"], 0)
        self.assertEqual(self.totals(), (2, 30))

    def test_invalid_precision_non_finite_and_negative_money(self):
        for value in [True, None, "invalid", -1, 0.001, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                to_cents(value)
        for value in [0, -1, 0.001, float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.refund(value, "bad")
        self.assertEqual(self.totals(), (0, 0))

    def test_draft_rejects_over_remaining_and_cross_customer(self):
        self.refund(200, "spent")
        for amount in [0, -1, 100, 999, 0.001]:
            draft = json.loads(tools.create_refund_draft.invoke({"order_id": "ORD-5001", "amount": amount}))
            self.assertEqual(draft["status"], "rejected")
        draft = json.loads(tools.create_refund_draft.invoke({"order_id": "ORD-5001", "amount": 99}))
        self.assertEqual(draft["status"], "drafted")
        self.assertEqual(draft["remaining_amount"], 99)
        tools.set_run_context("other-customer", "CUST-1002")
        denied = json.loads(tools.create_refund_draft.invoke({"order_id": "ORD-5001", "amount": 50}))
        self.assertEqual(denied["status"], "rejected")
        with self.assertRaises(ValueError):
            self.refund(50, "wrong-owner", customer="CUST-1002")

    def test_draft_does_not_accept_boolean_as_money(self):
        with self.assertRaises(ValueError):
            tools.create_refund_draft.invoke({"order_id": "ORD-5001", "amount": True})
        self.assertEqual(self.totals(), (0, 0))

    def test_request_identifier_is_required(self):
        with self.assertRaises(ValueError):
            self.refund(50, "")
        with self.assertRaises(TypeError):
            tools.issue_refund("ORD-5001", 50, customer_id="CUST-1001")

    def test_concurrent_distinct_requests_cannot_over_refund(self):
        barrier = threading.Barrier(2)
        def attempt(key):
            barrier.wait(timeout=5)
            try:
                return self.refund(200, key)["status"]
            except ValueError:
                return "rejected"
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, ["concurrent-one", "concurrent-two"]))
        self.assertCountEqual(results, ["refunded", "rejected"])
        self.assertEqual(self.totals(), (1, 20000))

    def test_concurrent_replay_executes_once(self):
        barrier = threading.Barrier(2)
        def attempt(_):
            barrier.wait(timeout=5)
            return self.refund(100, "same-concurrent")["status"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, range(2)))
        self.assertCountEqual(results, ["refunded", "already_refunded"])
        self.assertEqual(self.totals(), (1, 10000))

    def test_legacy_migration_preserves_data_and_is_repeatable(self):
        self.legacy_table()
        db.init_db()
        self.assertEqual(self.totals(), (1, 29900))
        repeated = self.refund(299, "legacy:RF-ORD-5001")
        self.assertEqual(repeated["status"], "already_refunded")
        self.refund(50, "new-one", "ORD-5002", "CUST-1002")
        self.refund(60, "new-two", "ORD-5002", "CUST-1002")
        db.init_db()
        self.assertEqual(self.totals(), (3, 40900))
        backups = list(self.database.parent.glob("*.before-partial-refunds-*.sqlite"))
        self.assertEqual(len(backups), 1)
        with closing(sqlite3.connect(backups[0])) as conn:
            self.assertEqual(conn.execute("SELECT amount FROM refunds").fetchone()[0], 299)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 5)

    def test_invalid_legacy_row_rolls_back_migration(self):
        self.legacy_table(amount=0)
        with self.assertRaises(sqlite3.IntegrityError):
            db.init_db()
        with closing(db.get_conn()) as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(refunds)")}
            self.assertNotIn("request_id", columns)
            self.assertEqual(conn.execute("SELECT amount FROM refunds").fetchone()[0], 0)
            self.assertIsNone(conn.execute("SELECT name FROM sqlite_master WHERE name='refunds_migration'").fetchone())

    def test_old_pending_action_reuses_migrated_refund(self):
        self.legacy_table()
        db.init_db()
        state = {"thread_id": "legacy-thread", "customer_id": "CUST-1001", "proposed_action": {
            "action": "issue_refund", "order_id": "ORD-5001", "amount": 299, "currency": "CNY",
        }}
        with patch.object(graph, "interrupt", return_value={"approved": True}):
            result = graph.approval_node(state)
        self.assertEqual(result["action_result"]["status"], "already_refunded")
        self.assertEqual(self.totals(), (1, 29900))

    def test_approval_can_adjust_amount_and_preserves_audit(self):
        app = self.workflow([200])
        approval_id = self.start(app, "adjust")
        self.assertEqual(self.totals(), (0, 0))
        result = graph.resume_turn("adjust", True, app=app, amount=50, approval_id=approval_id)
        self.assertEqual(result["state"]["action_result"]["amount"], 50)
        self.assertIn("¥50.00", result["state"]["final_reply"])
        payload = db.get_approval(approval_id)["payload"]
        self.assertEqual(payload["requested_amount"], 200)
        self.assertEqual(payload["approved_amount"], 50)
        self.assertEqual(payload["amount"], 50)
        with closing(db.get_conn()) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM actions_log WHERE action='approval_amount_adjusted'").fetchone()[0], 1)
        self.assertEqual(self.totals(), (1, 5000))

    def test_approval_rechecks_balance_after_another_refund(self):
        app = self.workflow([200, 200])
        first = self.start(app, "first-approval")
        second = self.start(app, "second-approval")
        graph.resume_turn("first-approval", True, app=app, approval_id=first)
        result = graph.resume_turn("second-approval", True, app=app, approval_id=second)
        self.assertEqual(result["state"]["approval_decision"], "failed")
        self.assertIn("未能执行", result["state"]["final_reply"])
        self.assertEqual(db.get_approval(second)["status"], "failed")
        self.assertEqual(self.totals(), (1, 20000))

    def test_separate_turns_in_same_thread_are_separate_requests(self):
        app = self.workflow([100, 50])
        first = self.start(app, "same-thread")
        graph.resume_turn("same-thread", True, app=app, approval_id=first)
        second = self.start(app, "same-thread")
        self.assertNotEqual(first, second)
        graph.resume_turn("same-thread", True, app=app, approval_id=second)
        self.assertEqual(self.totals(), (2, 15000))

    def test_pending_approval_survives_sqlite_checkpoint_reopen(self):
        # 关闭并重建工作流，验证申请编号不是依靠进程内变量保存的。
        self.workflow([75])
        checkpoint = self.database.parent / "checkpoints.sqlite"
        with patch.object(config, "CHECKPOINT_DB_PATH", checkpoint):
            first_app = graph.get_app()
            approval_id = self.start(first_app, "persistent-approval")
            request_id = db.get_approval(approval_id)["payload"]["request_id"]
            first_app.checkpointer.conn.close()
            restored_app = graph.get_app()
            try:
                result = graph.resume_turn("persistent-approval", True, app=restored_app,
                                           amount=50, approval_id=approval_id)
                self.assertEqual(result["state"]["action_result"]["request_id"], request_id)
                self.assertEqual(result["state"]["action_result"]["amount"], 50)
                self.assertEqual(self.totals(), (1, 5000))
            finally:
                restored_app.checkpointer.conn.close()

    def test_review_retry_cannot_create_another_refund(self):
        app = self.workflow([100, 50], reviews=[
            {"review": {"approved": False}, "retries": 1, "reviewer_feedback": "修正答复"},
            {"review": {"approved": True}},
        ])
        approval_id = self.start(app, "review-retry")
        result = graph.resume_turn("review-retry", True, app=app, approval_id=approval_id)
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["state"]["approval_decision"], "approved")
        self.assertEqual(self.totals(), (1, 10000))

    def test_reject_does_not_consume_balance(self):
        app = self.workflow([100])
        approval_id = self.start(app, "reject")
        graph.resume_turn("reject", False, app=app, approval_id=approval_id)
        self.assertEqual(db.get_approval(approval_id)["status"], "rejected")
        self.assertEqual(self.totals(), (0, 0))

    def test_review_exhaustion_keeps_executed_refund_receipt(self):
        app = self.workflow([100, 50], reviews=[
            {"review": {"approved": False}, "retries": 1},
            {"review": {"approved": False}, "retries": graph.MAX_REVIEW_RETRIES},
        ])
        approval_id = self.start(app, "review-exhaustion")
        result = graph.resume_turn("review-exhaustion", True, app=app, approval_id=approval_id)
        self.assertTrue(result["state"]["escalated"])
        self.assertIn("¥100.00", result["state"]["final_reply"])
        self.assertIn("完成模拟执行", result["state"]["final_reply"])
        self.assertIn("人工跟进", result["state"]["final_reply"])
        self.assertEqual(self.totals(), (1, 10000))

    def test_stale_approval_identifier_cannot_resume_another_request(self):
        app = self.workflow([50])
        approval_id = self.start(app, "stale")
        with self.assertRaisesRegex(ValueError, "暂停位置"):
            graph.resume_turn("stale", True, app=app, approval_id=approval_id+999)
        self.assertEqual(self.totals(), (0, 0))

    def test_staff_api_adjustment_and_repeated_submission(self):
        app = self.workflow([200])
        approval_id = self.start(app, "api-adjust")
        headers = {"X-Staff-Token": "test-staff-token"}
        with patch.object(api, "_workflow", return_value=app), patch.object(config, "STAFF_API_TOKEN", "test-staff-token"), TestClient(api.app) as client:
            url = f"/api/v1/staff/approvals/{approval_id}/decision"
            response = client.post(url, headers=headers, json={"approved": True, "staff_id": "test-staff", "amount": 50})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["status"], "approved")
            self.assertEqual(client.post(url, headers=headers, json={"approved": True, "staff_id": "test-staff", "amount": 50}).status_code, 409)
        self.assertEqual(self.totals(), (1, 5000))

    def test_staff_api_invalid_amount_is_rejected_before_execution(self):
        headers = {"X-Staff-Token": "test-staff-token"}
        with patch.object(config, "STAFF_API_TOKEN", "test-staff-token"), TestClient(api.app) as client:
            for value in [0, -1, 0.001, True, "NaN", "Infinity"]:
                response = client.post("/api/v1/staff/approvals/1/decision", headers=headers,
                                       json={"approved": True, "staff_id": "test-staff", "amount": value})
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.totals(), (0, 0))

    def test_staff_api_reports_execution_failure_instead_of_approved(self):
        app = self.workflow([50])
        approval_id = self.start(app, "api-over-limit")
        with patch.object(api, "_workflow", return_value=app), patch.object(config, "STAFF_API_TOKEN", "test-staff-token"), TestClient(api.app) as client:
            response = client.post(f"/api/v1/staff/approvals/{approval_id}/decision",
                                   headers={"X-Staff-Token": "test-staff-token"},
                                   json={"approved": True, "staff_id": "test-staff", "amount": 999})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "failed")
            self.assertEqual(response.json()["approval_decision"], "failed")
        self.assertEqual(self.totals(), (0, 0))

    def test_index_only_refresh_does_not_reset_business_data(self):
        self.refund(50, "keep-data")
        with patch("sys.argv", ["seed", "--index-only"]), \
             patch.object(seed, "seed_chroma") as refresh, redirect_stdout(io.StringIO()):
            seed.main()
        refresh.assert_called_once()
        self.assertEqual(self.totals(), (1, 5000))


if __name__ == "__main__":
    unittest.main()
