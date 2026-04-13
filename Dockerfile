# PedFormer-Enhanced Docker 訓練環境
# 基於 NVIDIA CUDA 12.1 + cuDNN 9 + Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 避免互動式安裝卡住
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 設定 python3 為預設
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 設定工作目錄
WORKDIR /workspace/PedFormer

# 先複製 requirements 以利用 Docker layer cache
COPY requirements.txt .

# 安裝 PyTorch (CUDA 12.1) 和其他依賴
RUN pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY configs/ configs/
COPY data/*.py data/
COPY data/__init__.py data/
COPY models/ models/
COPY losses/ losses/
COPY utils/ utils/
COPY scripts/ scripts/
COPY train.py evaluate.py inference.py ./

# 建立必要目錄
RUN mkdir -p weights logs data/flow_cache data/sam_cache

# 預設啟動訓練
CMD ["python", "train.py"]
