# RCPR-Identify

RCPR-Identify 是一个基于 SCoSPARC / RCPR 思路封装的多图共显著物体识别项目。用户上传至少 2 张包含共同目标的图片后，系统会调用训练好的模型进行共显著目标检测，并在网页中展示原图、Mask 和叠加可视化结果。

## 项目内容

- `前端/code.html`：TailwindCSS 前端页面，支持多图上传、预览、提交和结果展示。
- `cosod_backend.py`：轻量后端服务，提供 `/api/cosod` 接口，接收 `multipart/form-data` 的 `files[]` 字段并调用模型推理。
- `test.py` / `models.py` / `rcpr.py` / `dataset.py` / `utils.py`：SCoSPARC/RCPR 推理主链路。
- `start_cosod_web.bat`：Windows 一键启动脚本。
- `stop_cosod_web.bat`：Windows 一键停止脚本。
- `vis_compare/`：可视化对比产物。

## 权重说明

仓库只保留当前网页推理实际使用的权重：

- `checkpoints/baseline运行出的checkpoints/model_combo_base8-136_0.7291838924090067.pt`
- `models/dino_vitbase8_pretrain.pth`

其中 DINO 预训练权重较大，使用 Git LFS 管理。

## 数据与预测结果说明

`datasets/` 和 `predictions/` 目录不上传到 GitHub：

- 本项目网页功能通过上传图片临时推理，不依赖仓库内预置数据集。
- 如需运行原始测试脚本，请在本地自行放置 `datasets/CoCA`、`datasets/Cosal2015`、`datasets/CoSOD3k` 等数据。
- `predictions/` 是推理输出目录，运行后会在本地自动生成。

## 环境要求

本项目当前使用的 Conda 环境路径为：

```powershell
D:\Anaconda\Anaconda\envs\scosparc
```

主要依赖见：

- `requirements.txt`
- `self_requirements.txt`

## 启动网页应用

在 Windows 中双击项目根目录下的：

```text
start_cosod_web.bat
```

脚本会自动启动后端并打开：

```text
http://127.0.0.1:8765/code.html
```

关闭服务时双击：

```text
stop_cosod_web.bat
```

## 手动启动

也可以在 CMD 或 PowerShell 中运行：

```powershell
cd C:\COSOD\SCoSPARC-main课设版 && D:\Anaconda\Anaconda\envs\scosparc\python.exe cosod_backend.py --host 127.0.0.1 --port 8765
```

## 使用流程

1. 打开网页。
2. 上传至少 2 张包含共同目标的图片。
3. 点击“开始体验”。
4. 等待模型推理完成。
5. 查看原图、Mask 和叠加效果。

## 备注

该项目面向课程展示和本地运行场景，默认使用本地路径与 Windows 脚本；如需部署到服务器，需要调整 Conda 环境路径、模型权重路径和静态资源访问方式。
