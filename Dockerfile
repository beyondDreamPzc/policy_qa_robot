# ===== 构建阶段 =====
FROM python:3.11-slim AS builder

WORKDIR /app

# 先复制依赖清单，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== 运行阶段 =====
FROM python:3.11-slim

WORKDIR /app

# 从 builder 复制已装好的依赖
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码与数据
COPY config.py .
COPY app/ ./app/
COPY data/ ./data/

# 对外端口（uvicorn 监听 8000）
EXPOSE 8000

# 启动 Web 服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
