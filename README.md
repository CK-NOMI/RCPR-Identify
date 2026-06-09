# RCPR-Identify

RCPR-Identify 是一个“共显著物体识别平台”的前端项目。用户上传至少 2 张包含共同物体的图片后，前端会把图片提交到后端模型接口，并展示每张图片的原图、Mask 掩码图和叠加可视化图。

当前仓库已整理为 Cloudflare Pages 可直接部署的纯静态前端项目，不包含模型权重、训练数据集或推理输出结果。

## 项目结构

```text
/
├─ index.html
├─ assets/
├─ README.md
└─ .gitignore
```

## 功能说明

- 初始状态不显示任何示例图片。
- 支持点击上传和拖拽上传图片。
- 至少需要上传 2 张图片，否则提示“请至少上传 2 张包含共同物体的图片”。
- 支持 JPG、PNG 格式，单张图片默认限制 20MB。
- 上传后显示本地图片预览，并支持移除和清空。
- 点击“开始体验”后，以 `multipart/form-data` 提交到后端接口。
- 文件字段名为 `files`。
- 识别中按钮显示为“识别中...”并禁用。
- 识别完成后展示序号、原图、Mask 掩码、叠加效果和下载按钮。
- 兼容后端返回图片 URL，也预留 base64 图片返回格式。

## 本地运行方式

这是纯静态 HTML 项目，可以直接打开 `index.html`，也可以使用任意静态服务器预览。

例如使用 Python 启动本地静态服务：

```bash
python -m http.server 8080
```

然后访问：

```text
http://127.0.0.1:8080
```

## 后端接口说明

前端请求地址在 `index.html` 的 `<script>` 顶部配置：

```javascript
const API_BASE_URL = 'https://你的后端地址';
const COSOD_API = `${API_BASE_URL.replace(/\/$/, '')}/api/cosod`;
```

前端会向 `COSOD_API` 发起：

```text
POST /api/cosod
Content-Type: multipart/form-data
字段名：files
```

推荐后端返回格式：

```json
{
  "success": true,
  "results": [
    {
      "filename": "test_001.jpg",
      "image_url": "https://example.com/original/test_001.jpg",
      "mask_url": "https://example.com/mask/test_001.png",
      "overlay_url": "https://example.com/overlay/test_001.png"
    }
  ]
}
```

也兼容 base64 返回：

```json
{
  "success": true,
  "results": [
    {
      "filename": "test_001.jpg",
      "image_base64": "...",
      "mask_base64": "...",
      "overlay_base64": "..."
    }
  ]
}
```

注意：如果前端部署在 Cloudflare Pages，后端需要允许跨域请求，即配置 CORS 允许 Cloudflare Pages 域名访问 `/api/cosod`。

## Cloudflare Pages 部署方式

本项目是纯静态 HTML 项目，不需要安装依赖，也不需要构建。

Cloudflare Pages 设置：

```text
Build command: 留空
Build output directory: /
Root directory: /
```

如果 Cloudflare 页面显示 “Deploy command: npx wrangler deploy”，通常是创建成了 Workers 项目；请创建或切换为 Pages 静态站点部署。

## 注意事项

- 不要把模型权重提交到 GitHub，例如 `.pth`、`.pt`、`.ckpt`、`.onnx`。
- 不要把数据集提交到 GitHub，例如 `datasets/`、`dataset/`、`data/`。
- 不要把推理输出、上传缓存或结果目录提交到 GitHub，例如 `outputs/`、`results/`、`uploads/`、`predictions/`。
- Cloudflare Pages 只能部署前端静态页面，不能直接运行 Python 深度学习模型。模型推理需要部署在独立后端服务中，前端通过 `API_BASE_URL` 调用它。
