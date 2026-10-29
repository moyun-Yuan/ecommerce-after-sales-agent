"""订单权限、退款幂等和员工接口鉴权测试。"""
from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from ecommerce_after_sales_agent import config, db, seed, tools


class SecurityAndApiTests(unittest.TestCase):
    """验证由程序强制执行的业务边界。"""

    def setUp(self) -> None:
        cache_root = Path(__file__).resolve().parents[1] / ".cache"
        cache_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="security-test-", dir=cache_root)
        self.database = Path(self.temporary.name) / "business.sqlite"
        self.db_patch = patch.object(config, "DB_PATH", self.database)
        self.db_patch.start()
        seed.seed_sqlite()

    def tearDown(self) -> None:
        self.db_patch.stop()
        gc.collect()
        self.temporary.cleanup()

    def test_order_tools_reject_cross_customer_access(self):
        """张伟不能通过工具读取林晓的订单。"""
        tools.set_run_context("thread-access", "CUST-1002")
        result = json.loads(tools.get_order.invoke({"order_id": "ORD-5001"}))
        self.assertIn("无权", result["error"])

    def test_refund_draft_validates_paid_amount(self):
        """退款草案金额必须与当前客户的订单实付金额一致。"""
        tools.set_run_context("thread-amount", "CUST-1001")
        rejected = json.loads(
            tools.create_refund_draft.invoke({"order_id": "ORD-5001", "amount": 999})
        )
        accepted = json.loads(
            tools.create_refund_draft.invoke({"order_id": "ORD-5001", "amount": 299})
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(accepted["status"], "drafted")

    def test_refund_execution_is_idempotent(self):
        """同一订单重复执行时复用已有退款记录，不产生第二笔退款。"""
        first = tools.issue_refund("ORD-5001", 299, "thread-first", "CUST-1001")
        second = tools.issue_refund("ORD-5001", 299, "thread-second", "CUST-1001")
        self.assertEqual(first["status"], "refunded")
        self.assertEqual(second["status"], "already_refunded")
        with db.get_conn() as conn:
            refund_count = conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]
            issue_count = conn.execute(
                "SELECT COUNT(*) FROM actions_log WHERE action = 'issue_refund'"
            ).fetchone()[0]
        self.assertEqual(refund_count, 1)
        self.assertEqual(issue_count, 1)

    def test_staff_api_requires_token(self):
        """员工审批列表不能匿名访问。"""
        with patch.object(config, "STAFF_API_TOKEN", "test-staff-token"):
            with TestClient(api.app) as client:
                unauthorized = client.get("/api/v1/staff/approvals")
                authorized = client.get(
                    "/api/v1/staff/approvals",
                    headers={"X-Staff-Token": "test-staff-token"},
                )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertEqual(authorized.json()["count"], 0)


if __name__ == "__main__":
    unittest.main()
