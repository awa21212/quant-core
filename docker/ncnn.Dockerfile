# ncnn 构建镜像（Ubuntu 24.04 + pybind11 + 可选 libtorch）
#
# 为什么用它：宿主机上没有 root、缺 python 头文件、`-j20` 会被 OOM kill。
# 容器里有 root、内存有上限、路径是 Linux 原生路径，这些问题都不存在。
#
# 构建（基础版，只编 ncnn 的 python 模块）：
#   docker build -t ncnn-build:24.04 -f docker/ncnn.Dockerfile docker
#
# 构建（带 torch，能编 pnnx）：
#   docker build -t ncnn-build:24.04-torch \
#     --build-arg WITH_TORCH=1 -f docker/ncnn.Dockerfile docker

FROM ubuntu:24.04

ARG WITH_TORCH=0
ARG TORCH_VER=2.15.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        ca-certificates \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# pybind11 / numpy / pytest：ncnn 的 python 绑定与测试都要用
RUN pip3 install --no-cache-dir --break-system-packages \
        "pybind11>=2.12" numpy pytest

# 可选：libtorch（编 pnnx 需要）。这会把镜像撑到 4–5 GB，按需开启。
RUN if [ "$WITH_TORCH" = "1" ]; then \
        pip3 install --no-cache-dir --break-system-packages \
            --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VER}"; \
    fi

WORKDIR /work
CMD ["bash"]
