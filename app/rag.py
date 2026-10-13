# -*- coding: utf-8 -*-
"""
RAG 核心逻辑：embed_query（向量化查询）→ search（Milvus 检索）→ generate（LLM 生成）。

对外主要入口：ask(question) -> {answer, sources}
其中 sources 为命中的参考问答对列表（满足「回答标注参考内容」要求）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

import requests  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402


_client = None


def get_client() -> MilvusClient:
    """惰性获取 Milvus 客户端（复用连接）。"""
    global _client
    if _client is None:
        _client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")
    return _client


def embed_query(text: str):
    """用 bge-m3 把查询文本转成向量。"""
    url = f"{config.OLLAMA_BASE_URL}/api/embed"
    resp = requests.post(url, json={"model": config.EMBED_MODEL, "input": [text]}, timeout=120)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def search(query: str, top_k: int = config.TOP_K):
    """Milvus 向量检索，返回命中的参考问答对列表。

    返回：list[dict]，每项含 id/question/answer/score。
    """
    client = get_client()
    vec = embed_query(query)
    results = client.search(
        collection_name=config.COLLECTION_NAME,
        data=[vec],
        limit=top_k,
        output_fields=["id", "question", "answer"],
        search_params=config.SEARCH_PARAM,
    )
    hits = results[0] if results else []
    sources = []
    for h in hits:
        entity = h.get("entity", {})
        sources.append({
            "id": entity.get("id"),
            "question": entity.get("question", ""),
            "answer": entity.get("answer", ""),
            "score": round(h.get("distance", 0.0), 4),
        })
    return sources


def generate(query: str, sources):
    """用 deepseek-r1:1.5b 基于召回上下文生成答案。

    失败或超时时降级为返回 top-1 原文答案。
    """
    if not sources:
        return "抱歉，知识库中未找到与您问题相关的政策内容，请换个问法试试。"

    context_text = "\n\n".join(
        f"[{i+1}] 问题：{s['question']}\n答案：{s['answer']}"
        for i, s in enumerate(sources)
    )
    prompt = (
        "你是宝安区政策咨询助手。请仅依据下面提供的政策问答参考内容，"
        "用简洁、准确的中文回答用户的问题，不要编造知识库以外的信息。\n\n"
        f"参考内容：\n{context_text}\n\n"
        f"用户问题：{query}\n\n"
        "请直接给出回答："
    )

    url = f"{config.OLLAMA_BASE_URL}/api/generate"
    try:
        resp = requests.post(
            url,
            json={
                "model": config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 256},
            },
            timeout=config.LLM_TIMEOUT,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        if answer:
            return answer
    except Exception as e:  # noqa: BLE001
        print(f"[rag] LLM 生成失败，降级为 top-1 原文：{e}", file=sys.stderr)

    # 降级：直接返回最相关的原文答案
    return sources[0]["answer"]


def ask(question: str, top_k: int = config.TOP_K):
    """完整 RAG 流程入口。返回 {answer, sources}。"""
    question = (question or "").strip()
    if not question:
        return {"answer": "请输入您想咨询的政策问题。", "sources": []}

    sources = search(question, top_k=top_k)
    answer = generate(question, sources)
    return {"answer": answer, "sources": sources}


def is_ready() -> bool:
    """判断 collection 是否已建好（供 Web 服务启动时提示）。"""
    try:
        return get_client().has_collection(config.COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        return False
