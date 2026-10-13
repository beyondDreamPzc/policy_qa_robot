# -*- coding: utf-8 -*-
"""
全局配置：Milvus / Ollama 地址、collection 名、模型名、检索参数等。
所有模块从这里读取统一常量，避免散落魔法值。
"""
import os

# ---------- 路径 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = os.path.join(BASE_DIR, "data", "knowledgeBase.xlsx")   # 源数据
DATA_JSONL = os.path.join(BASE_DIR, "data", "qa.jsonl")            # 抽取后的问答对
EMBED_CACHE_DIR = os.path.join(BASE_DIR, "data", ".embed_cache")   # 批次级向量缓存

# ---------- Milvus ----------
# 地址/端口支持环境变量覆盖（Docker 容器内需用服务名而非 127.0.0.1）
MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "policy_qa")
EMBED_DIM = 1024          # bge-m3 输出维度
INDEX_PARAM = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}
SEARCH_PARAM = {"metric_type": "COSINE", "params": {"nprobe": 16}}
TOP_K = 5                  # 召回数量

# ---------- Ollama ----------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3:latest")          # embedding 模型
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-r1:1.5b")           # 生成模型
EMBED_BATCH_SIZE = 16                  # 每批向量化条数（bge-m3 CPU 一次批量送）
LLM_TIMEOUT = 120                      # 生成超时（秒）

# ---------- 其它 ----------
DEFAULT_LIMIT = 200    # 采样建库默认条数；0 表示全量
