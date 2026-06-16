FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# 安装系统依赖 + Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        libglib2.0-0 \
        libgl1 \
        tini \
        bash && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# 复制并安装 Python 依赖（重点优化 PyTorch）
COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    # 强制安装 CUDA 兼容的 PyTorch（cu121 wheel 在 CUDA 12.4 runtime 上可正常运行）
    python3 -m pip install --no-cache-dir \
        torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 复制代码
COPY models/ /app/models/
#COPY src/ /app/src/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/app/run.sh"]