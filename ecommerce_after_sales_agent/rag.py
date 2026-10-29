"""RAG 检索：Chroma 向量召回、词项混合重排和政策来源编号。

构建索引时先切分政策、生成向量并写入 Chroma；查询时召回 ``fetch_k`` 个候选，
再融合 BGE 余弦相似度与 BM25 风格词项分数，保留前 ``k`` 个结果。
"""
from __future__ import annotations

import math
import re
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions

from . import config
from .kb_docs import KB_DOCS, KB_ZH, POLICY_VERSION

_client = None
_collection = None
_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
    return _embed_fn


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def _get_collection():
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=_get_embed_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _chunk(text: str, max_chars: int = 350) -> list[str]:
    """按句子组合文本，每个块尽量不超过 max_chars 个字符。"""
    sentences = [
        sentence for sentence in re.split(r"(?<=[。！？.!?])\s*", text.strip())
        if sentence
    ]
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > max_chars:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf.strip())
    return chunks


def _zh_tokens(text: str) -> list[str]:
    """生成单字、双字词组及英文数字词，用于小语料 BM25 重排。"""
    normalized = text.lower()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    for segment in re.findall(r"[\u3400-\u9fff]+", normalized):
        tokens.extend(segment)
        tokens.extend(segment[index:index + 2] for index in range(len(segment) - 1))
    return tokens


def _normalize(values: list[float]) -> list[float]:
    """把一组分数缩放到 0 至 1；单个候选直接记为 1。"""
    if len(values) <= 1:
        return [1.0] * len(values)
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def _hybrid_scores(query: str, docs: list[str], distances: list[float]) -> list[float]:
    """融合 BGE 余弦相似度和 BM25 风格词项分数。"""
    tokenized_docs = [_zh_tokens(doc) for doc in docs]
    query_tokens = set(_zh_tokens(query))
    doc_count = len(tokenized_docs)
    average_length = sum(map(len, tokenized_docs)) / max(doc_count, 1)
    document_frequency = Counter(
        token for tokens in tokenized_docs for token in set(tokens)
    )
    lexical_scores: list[float] = []
    k1, b = 1.5, 0.75
    for tokens in tokenized_docs:
        frequencies = Counter(tokens)
        length_ratio = len(tokens) / max(average_length, 1.0)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (doc_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            score += inverse_frequency * (
                frequency * (k1 + 1)
                / (frequency + k1 * (1 - b + b * length_ratio))
            )
        lexical_scores.append(score)

    # Chroma 的 cosine distance 通常位于 0 至 2；直接映射可以保留候选间的真实差距，
    # 避免 min-max 把极小的距离差异夸大成 0 和 1。
    vector_scores = [
        max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
        for distance in distances
    ]
    normalized_lexical = _normalize(lexical_scores)
    return [
        round(0.65 * vector + 0.35 * lexical, 6)
        for vector, lexical in zip(vector_scores, normalized_lexical)
    ]


def build_index() -> int:
    """使用 KB_DOCS 重建 Chroma 集合并返回文本块数量。"""
    client = _get_client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
    # 删除集合后清空模块级引用，确保重新创建空集合。
    global _collection
    _collection = None
    col = _get_collection()

    ids, docs, metas = [], [], []
    for doc in KB_DOCS:
        for i, chunk in enumerate(_chunk(doc["text"])):
            ids.append(f'{doc["id"]}::{i}')
            docs.append(chunk)
            metas.append({"doc_id": doc["id"], "title": doc["title"], "chunk": i,
                          "policy_version": POLICY_VERSION, "currency": "CNY"})
    col.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def retrieve(query: str, k: int = 3, fetch_k: int = 8) -> list[dict]:
    """召回并重排文本，返回前 k 条内容及其来源编号。"""
    search_query = query.strip()
    if not search_query:
        return []
    embedding_query = config.EMBEDDING_QUERY_INSTRUCTION + search_query
    col = _get_collection()
    res = col.query(
        query_texts=[embedding_query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    if not docs:
        return []

    # 检索指令只服务于 BGE 查询向量，词项重排仍使用原始问题。
    scores = _hybrid_scores(search_query, docs, distances)
    ranked = sorted(
        zip(scores, docs, metas), key=lambda x: float(x[0]), reverse=True
    )[:k]
    return [
        {
            "doc_id": m["doc_id"],
            "title": KB_ZH.get(m["doc_id"], {}).get("title", m["title"]),
            "text": KB_ZH.get(m["doc_id"], {}).get("text", d),
            "source_text": d,
            "search_query": search_query,
            "rerank_method": config.RERANK_METHOD,
            "policy_version": POLICY_VERSION,
            "currency": "CNY",
            "score": round(float(s), 4),
        }
        for s, d, m in ranked
    ]


if __name__ == "__main__":
    # 命令行快速检查：python -m ecommerce_after_sales_agent.rag "我的包裹一直没收到"
    import sys

    q = " ".join(sys.argv[1:]) or "我的包裹一直没收到，可以退款吗？"
    print(f"检索问题：{q}\n")
    for r in retrieve(q):
        print(f"[{r['doc_id']}] ({r['score']})  {r['title']}")
        print(f"    {r['text'][:160]}...\n")
