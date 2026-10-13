# -*- coding: utf-8 -*-
"""
FastAPI 后端服务：提供 RAG 问答接口与前端页面。

启动：
    uvicorn app.main:app --host 127.0.0.1 --port 8000
接口：
    GET  /          -> 前端对话页面
    GET  /api/health -> 健康检查（返回 collection 是否就绪）
    POST /api/ask    -> {"question": "..."} -> {"answer": "...", "sources": [...]}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.rag import ask, is_ready  # noqa: E402


app = FastAPI(title="宝安区政策问答 RAG 机器人", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class AskRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K


class AskResponse(BaseModel):
    answer: str
    sources: list


@app.get("/api/health")
def health():
    return {"status": "ok", "collection_ready": is_ready(), "collection": config.COLLECTION_NAME}


@app.get("/favicon.ico")
def favicon():
    # 浏览器会自动请求 favicon，返回 204 避免控制台 404 噪音
    from fastapi import Response
    return Response(status_code=204)


@app.post("/api/ask")
def api_ask(req: AskRequest):
    result = ask(req.question, top_k=req.top_k)
    return AskResponse(answer=result["answer"], sources=result["sources"])


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# 静态资源挂载（css/js 等，若有）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
