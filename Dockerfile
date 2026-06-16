FROM python:3.11-slim

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app /saisresult && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        libglib2.0-0 \
        libgl1 \          # ← Changed from libgl1-mesa-glx
        tini \
        bash && \
    rm -rf /var/lib/apt/lists/*

# 将 python3.11 设为默认 python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .
RUN  pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY models/ /app/models/
#COPY src/ /app/src/
COPY run_inference.py /app/src/run_inference.py
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/sbin/tini", "--", "bash", "/app/run.sh"]