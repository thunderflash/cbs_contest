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
        
RUN wget -q -O /tmp/archive.zip "https://storage.googleapis.com/kaggle-data-sets/10868085/17116812/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260624%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260624T031751Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=b6b7d2cb9bf1add0f607ee964d71532ef5ba5beba768113088da6ab7bf7fe1f6d60385a505d0b439808e096d5a49f97188133f576027600a3491d2065c5afabd8d2df76014ad4f3af5f62dd57e8c8addb70699dc2ef3e776c7941121f9ec4ef5f6ef7be33029b1b23aadbd84c0c8cb5a29883ca74d21918b7daa7e27bbe1b983d389428a81949b40d2a305d7949217e00861f574e4516544279a788305f684d9ec246443e7914dfc81d1aa11401b63b042ced190a3592674d77337ac948fadf1e45f0a5dce6e03e844d4a7bff3fc28eebf80f75ded700d5c51259c08e25b5fe920e7565382371cbf89778d5e97e9818981bbfbedd6ae964db1061e6da63a0b42" && \
    unzip -q /tmp/archive.zip -d "/app/models/" && \
    rm /tmp/archive.zip

# 复制代码
#COPY models/ /app/models/
#COPY src/ /app/src/
COPY run.sh /app/run.sh
COPY run_inference.py /app/run_inference.py
RUN chmod +x /app/run.sh

ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/app/run.sh"]