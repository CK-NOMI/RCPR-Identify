# 后端部署说明

本目录用于在华为云 ECS 上部署共显著物体识别模型后端。Cloudflare 前端会请求同域 `/api/cosod`，再由 Worker 代理到：

```text
http://120.46.136.60:8080/api/cosod
```

## 1. 克隆仓库

```bash
cd /root
git clone https://github.com/CK-NOMI/RCPR-Identify.git
cd RCPR-Identify/backend
```

## 2. 创建 Python 虚拟环境

建议服务器使用 Python 3.10 或系统默认 Python 3.12。

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple
```

如果腾讯云镜像仍然较慢，可以换华为云 PyPI 镜像：

```bash
pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple --trusted-host repo.huaweicloud.com
```

`pydensecrf` 不是必装项。当前后端会在未安装 `pydensecrf` 时自动跳过 CRF 精修，先保证模型推理服务可部署、可调用。

## 3. 放置模型权重

权重不提交到 GitHub，需要手动上传到服务器。

业务 checkpoint 放到：

```text
backend/checkpoints/baseline运行出的checkpoints/model_combo_base8-136_0.7291838924090067.pt
```

DINO 预训练权重放到：

```text
backend/models/dino_vitbase8_pretrain.pth
```

可以先创建目录：

```bash
mkdir -p "checkpoints/baseline运行出的checkpoints" models
```

## 4. 启动后端

```bash
bash start_backend.sh
```

启动成功后，本机测试：

```bash
curl http://127.0.0.1:8080/api/health
```

公网测试：

```bash
curl http://120.46.136.60:8080/api/health
```

华为云安全组需要放行 TCP `8080` 端口。

## 5. 与前端的关系

前端地址：

```text
https://rcpr-identify.chenkang314.workers.dev/
```

前端请求：

```text
/api/cosod
```

Cloudflare Worker 代理：

```text
/api/cosod -> http://120.46.136.60:8080/api/cosod
/outputs/* -> http://120.46.136.60:8080/outputs/*
```

因此浏览器不会直接从 HTTPS 页面请求 HTTP 后端，避免 Mixed Content 拦截。
