"""实验收尾和离线报告恢复测试，不调用真实模型。"""
from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_thinking as benchmark
from ecommerce_after_sales_agent import config, db


class BenchmarkRecoveryTests(unittest.TestCase):
    def test_database_connection_closes_even_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "business.sqlite"
            original = db.get_conn
            with patch.object(config, "DB_PATH", path), benchmark.closed_database_connections():
                with self.assertRaisesRegex(RuntimeError, "模拟业务异常"):
                    with db.get_conn() as connection:
                        connection.execute("CREATE TABLE test (value INTEGER)")
                        connection.execute("INSERT INTO test VALUES (1)")
                        raise RuntimeError("模拟业务异常")
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")
            self.assertIs(db.get_conn, original)
            # Windows 下这一步也必须成功，证明强引用仍存在时连接已被显式关闭。
            path.unlink()

    def test_atomic_progress_also_generates_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / ".cache").mkdir()
            output = root / ".cache" / "thinking-benchmark-20261029.json"
            data = {"rows": [], "replications": []}
            with patch.object(benchmark, "ROOT", root):
                benchmark.save_progress(data, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), data)
            self.assertFalse(output.with_suffix(".json.tmp").exists())
            self.assertTrue((root / "docs" / "思考模式节点实测报告_20261029.md").exists())

    def test_report_only_requires_no_key_or_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / ".cache").mkdir()
            (root / ".cache" / "thinking-benchmark-20261029.json").write_text(
                json.dumps({"rows": [], "replications": []}), encoding="utf-8")
            with patch.object(benchmark, "ROOT", root), patch.object(config, "DEEPSEEK_API_KEY", ""), \
                 patch.object(benchmark, "create_chat_model", side_effect=AssertionError("不应调用模型")) as factory, \
                 patch("sys.argv", ["benchmark_thinking.py", "--report-only"]), contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            factory.assert_not_called()
            self.assertTrue((root / "docs" / "思考模式节点实测报告_20261029.md").exists())


if __name__ == "__main__":
    unittest.main()
