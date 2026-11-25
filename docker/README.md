# docker/ —— 复杂任务的隔离环境

宿主机上踩过的三个坑，这里都不存在：

| 坑 | 宿主机上 | 容器里 |
|---|---|---|
| 缺 python 头文件 | Ubuntu 无 root，只能 `dpkg -x` 扒 `.deb` | `apt install python3-dev` |
| `-j20` 被 OOM 杀掉 | 内存被 Windows 抢 | `--memory 8g` 硬上限，溢出就报错不会杀掉整机 |
| Windows 长路径装包失败 | `torch-2.15.0.dist-info\licenses\...` 超 260 字符 | Linux 路径无此限制 |

## 构建

```powershell
# 基础版（只编 ncnn python 模块，约 1.2 GB）
docker build -t ncnn-build:24.04 -f docker/ncnn.Dockerfile docker

# 带 libtorch（能编 pnnx，约 4–5 GB）
docker build -t ncnn-build:24.04-torch --build-arg WITH_TORCH=1 -f docker/ncnn.Dockerfile docker
```

## 使用

```powershell
.\docker\run.ps1 -Cmd "cmake --version"
.\docker\run.ps1 -Torch -Cmd "cd /work/ncnn && ./build_pnnx.sh"
```

挂载约定：

| 宿主 | 容器 | 用途 |
|---|---|---|
| `D:\ps\ncnn` | `/work/ncnn` | 源码（读写） |
| `D:\ps\build` | `/work/build` | 构建目录（避免污染源码） |
| `桌面\量化回测` | `/work/quant` | 脚本、测试文件 |

## 在容器里编 ncnn python 模块（已实测通过）

```powershell
.\docker\run.ps1 -Cmd 'cmake -S /work/ncnn -B /work/build/py \
  -DNCNN_PYTHON=ON -DNCNN_SYSTEM_PYBIND11=ON \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release && \
  cmake --build /work/build/py -j6'
```

两个坑（都已踩过并解决）：

1. 不加 `-Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)` 会报 `pybind11 package not found`，然后因为子模块没拉下来直接失败；
2. 编出来的 `.so` 在 `build/py/python/ncnn/` 里，`PYTHONPATH` 要指到**那一层**：
   ```bash
   export PYTHONPATH=/work/build/py/python/ncnn   # 不是 .../python
   python3 -m pytest /work/ncnn/python/tests/ -q
   ```

实测：993 MB 镜像、编译约 2 分钟（-j6）、`test_mat_buffer.py` 6 passed。

## 磁盘


Docker 的数据默认在 **C 盘**，而 C 只剩 ~48 GB。用完记得：

```powershell
docker system prune -f
docker image prune -f
```

想长期用，把 Docker Desktop 的 disk image location 迁到 D 盘（Settings → Resources）。

## 不解决什么

- GitHub 推送被墙 → 仍走 API 直传；
- 维护者 CLA 没回调 → 容器再干净也没用；
- 数据授权（东财/巨潮）→ 不能随镜像分发。
