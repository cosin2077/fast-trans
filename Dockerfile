# 使用官方Python基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装依赖
COPY src/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制源代码
COPY src/ .

# 暴露端口
EXPOSE 8000

# 运行命令
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8002", "--workers", "2"]