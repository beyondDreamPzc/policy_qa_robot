# -*- coding: utf-8 -*-
"""
建索引：读取 qa.jsonl，调用 Ollama bge-m3 生成向量，写入 Milvus collection。

用法（务必用 python -u，避免输出缓冲导致进度不刷新）：
    python -u scripts/build_index.py                     # 默认读 data/qa.jsonl（全量）
    python -u scripts/build_index.py --limit 200         # 只建前 200 条（采样）
    python -u scripts/build_index.py --drop              # 重建前先删旧 collection

特性：
    - 批次级缓存（data/.embed_cache/NNNNN.json），中断后重跑自动跳过已完成批次。
    - 写入前判断 collection 是否存在，避免重复建表。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

import requests  # noqa: E402
from pymilvus import (  # noqa: E402
    Collection, CollectionSchema, DataType, FieldSchema,
    connections, utility,
)


# ---------------- Ollama embedding ----------------

def embed_batch(texts):
    """调用 Ollama /api/embed，一次批量返回多条向量。"""
    url = f"{config.OLLAMA_BASE_URL}/api/embed"
    resp = requests.post(url, json={
        "model": config.EMBED_MODEL,
        "input": texts,
    }, timeout=600)
    resp.raise_for_status()
    return [e for e in resp.json()["embeddings"]]


# ---------------- 批次缓存 ----------------

def cache_path(batch_idx: int) -> str:
    os.makedirs(config.EMBED_CACHE_DIR, exist_ok=True)
    return os.path.join(config.EMBED_CACHE_DIR, f"{batch_idx:05d}.json")


def load_cache(batch_idx: int):
    p = cache_path(batch_idx)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(batch_idx: int, vectors):
    p = cache_path(batch_idx)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(vectors, f)
    os.replace(tmp, p)  # 原子写，避免中断写坏


# ---------------- Milvus ----------------

def ensure_collection(drop: bool = False) -> Collection:
    """不存在则创建 collection（含 index），返回 Collection 对象。"""
    connections.connect(alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)

    if drop and utility.has_collection(config.COLLECTION_NAME):
        utility.drop_collection(config.COLLECTION_NAME)
        print(f"[index] 已删除旧 collection '{config.COLLECTION_NAME}'")

    if utility.has_collection(config.COLLECTION_NAME):
        print(f"[index] collection '{config.COLLECTION_NAME}' 已存在，直接复用")
        return Collection(config.COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBED_DIM),
    ]
    schema = CollectionSchema(fields, description="宝安区政策问答知识库")
    col = Collection(config.COLLECTION_NAME, schema=schema)
    col.create_index(field_name="embedding", index_params=config.INDEX_PARAM)
    print(f"[index] 已创建 collection '{config.COLLECTION_NAME}' 并建立索引")
    return col


def main():
    parser = argparse.ArgumentParser(description="向量化 qa.jsonl 并写入 Milvus")
    parser.add_argument("--jsonl", type=str, default=config.DATA_JSONL, help="输入 jsonl")
    parser.add_argument("--limit", type=int, default=0, help="只建前 N 条，0 表示全部")
    parser.add_argument("--drop", action="store_true", help="重建前先删除已有 collection")
    args = parser.parse_args()

    with open(args.jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"[index] 待向量化 {len(rows)} 条")

    col = ensure_collection(drop=args.drop)
    col.load()

    total = len(rows)
    batch = config.EMBED_BATCH_SIZE
    inserted = 0

    for start in range(0, total, batch):
        end = min(start + batch, total)
        batch_idx = start // batch
        chunk = rows[start:end]

        vectors = load_cache(batch_idx)
        if vectors is None:
            texts = [r["question"] for r in chunk]
            vectors = embed_batch(texts)
            save_cache(batch_idx, vectors)

        entities = [
            [r["id"] for r in chunk],
            [r["question"] for r in chunk],
            [r["answer"] for r in chunk],
            vectors,
        ]
        col.insert(entities)
        inserted += len(chunk)
        print(f"[index] 进度 {inserted}/{total}（batch {batch_idx}）", flush=True)

    col.flush()
    print(f"[index] 完成：共写入 {inserted} 条 → collection '{config.COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
