"""显示格式、检索与重排、审批流程测试；不调用真实模型接口。"""
from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from ecommerce_after_sales_agent import config, db, graph, rag, seed
from ecommerce_after_sales_agent.i18n import customer_label, localize_order, money
from ecommerce_after_sales_agent.kb_docs import KB_ZH


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, *, method):
        """模拟 JSON 模式并执行真实 Pydantic 校验，不请求外部服务。"""
        return RunnableLambda(lambda messages: schema.model_validate_json(self.invoke(messages).content))

    def invoke(self, messages):
        return next(self.replies)


class ChineseFlowTests(unittest.TestCase):
    def test_display_keeps_ids_currency_and_custom_names(self):
        self.assertEqual(customer_label("CUST-1001", "林晓"), "林晓")
        self.assertEqual(customer_label("CUST-1001", "我的测试客户"), "我的测试客户")
        order = localize_order({"id": "ORD-5001", "item": "无线头戴耳机", "status": "lost", "amount": 299.00})
        self.assertEqual(order["item"], "无线头戴耳机")
        self.assertEqual(order["status_label"], "运输丢失")
        self.assertEqual(order["status"], "lost")
        self.assertEqual(order["amount"], 299.00)
        self.assertEqual(order["currency"], "CNY")
        self.assertEqual(money(order["amount"]), "¥299.00（人民币）")

    def test_chinese_query_is_used_directly_for_retrieval_and_reranking(self):
        query = "我的包裹丢失了，能退款吗？"
        collection = Mock()
        collection.query.return_value = {
            "documents": [["承运商确认丢件后，可以按订单实付金额提出退款方案。"]],
            "metadatas": [[{"doc_id": "refund-policy", "title": "人民币退款与丢件处理"}]],
            "distances": [[0.1]],
        }
        with patch.object(rag, "_get_collection", return_value=collection):
            results = rag.retrieve(query)
        collection.query.assert_called_once_with(
            query_texts=[config.EMBEDDING_QUERY_INSTRUCTION + query],
            n_results=8,
            include=["documents", "metadatas", "distances"],
        )
        self.assertEqual(results[0]["title"], "人民币退款与丢件处理")
        self.assertIn("丢失", results[0]["text"])
        self.assertIn("丢件", results[0]["source_text"])
        self.assertEqual(results[0]["rerank_method"], "bge_cosine_plus_zh_bm25")

    def test_chinese_hybrid_reranker_uses_lexical_evidence(self):
        query = "定制商品可以七天无理由退货吗"
        docs = [
            "快递配送时间通常为二至五天。",
            "消费者定作的商品可能不适用七日无理由退货。",
        ]
        scores = rag._hybrid_scores(query, docs, [0.20, 0.21])
        self.assertGreater(scores[1], scores[0])

    def test_approval_and_rejection_keep_chinese_reply_and_execution_boundary(self):
        cache_root = Path(__file__).resolve().parents[1] / ".cache"
        cache_root.mkdir(exist_ok=True)
        for approved in (True, False):
            with self.subTest(approved=approved):
                temporary = tempfile.TemporaryDirectory(prefix="zh-test-", dir=cache_root)
                database = Path(temporary.name) / "business.sqlite"
                self.assertTrue(database.resolve().is_relative_to(cache_root.resolve()))
                def tool_call(name, args, call_id):
                    return {"name": name, "args": args, "id": call_id}
                model = ScriptedModel([
                    AIMessage(content=json.dumps({"route": "use_tools", "reason": "客户申请丢失订单退款"}, ensure_ascii=False)),
                    AIMessage(content="", tool_calls=[tool_call("get_order", {"order_id": "ORD-5001"}, "order")]),
                    AIMessage(content="", tool_calls=[tool_call("get_tracking", {"order_id": "ORD-5001"}, "tracking")]),
                    AIMessage(content="", tool_calls=[tool_call("check_refund_policy", {"query": "lost package refund"}, "policy")]),
                    AIMessage(content="", tool_calls=[tool_call("create_refund_draft", {"order_id": "ORD-5001", "amount": 299.00}, "draft")]),
                    AIMessage(content="已确认包裹丢失，299元人民币退款草案等待员工审核。[refund-policy]"),
                    AIMessage(content='{"approved": true, "grounded": true, "citations_ok": true, "pii_found": false, "reason": "政策依据和引用有效"}'),
                ])
                with patch.object(config, "DB_PATH", database), \
                     patch.object(graph, "_llm", return_value=model), \
                     patch.object(rag, "retrieve", return_value=[{
                         "doc_id": "refund-policy", "title": "退款政策", "text": KB_ZH["refund-policy"]["text"], "score": 2.0,
                     }]):
                    seed.seed_sqlite()
                    app = graph.build_graph(InMemorySaver())
                    thread = f"chinese-test-{approved}"
                    result = graph.run_turn("订单ORD-5001没收到，我想退款", "CUST-1001", thread, app=app)
                    self.assertEqual(result["status"], "interrupted")
                    with db.get_conn() as connection:
                        self.assertEqual(connection.execute("SELECT COUNT(*) FROM actions_log WHERE action = 'issue_refund'").fetchone()[0], 0)
                    connection.close()
                    result = graph.resume_turn(thread, approved=approved, app=app)
                    reply = result["state"]["final_reply"]
                    self.assertIn("政策来源：人民币退款与丢件处理", reply)
                    self.assertIn("模拟执行" if approved else "未获批准", reply)
                    self.assertNotIn("美元", reply)
                    self.assertNotIn("等待员工审核", reply)
                    self.assertNotIn("尚未执行", reply)
                    if approved:
                        self.assertEqual(result["state"]["action_result"]["currency"], "CNY")
                    with db.get_conn() as connection:
                        self.assertEqual(connection.execute("SELECT COUNT(*) FROM actions_log WHERE action = 'issue_refund'").fetchone()[0], int(approved))
                    connection.close()
                del app, result, model
                gc.collect()
                temporary.cleanup()

    def test_domestic_policy_terms_are_consistent(self):
        policies = " ".join(doc["text"] for doc in KB_ZH.values())
        for expected in ("人民币", "七日", "支付宝", "微信支付", "北京时间", "顺丰"):
            self.assertIn(expected, policies)
        for obsolete in ("美元", "美东", "UPS", "FedEx", "USPS", "30天", "5至7个工作日"):
            self.assertNotIn(obsolete, policies)


if __name__ == "__main__":
    unittest.main()
