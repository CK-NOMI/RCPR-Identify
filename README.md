# RCPR-Identify

RCPR-Identify 是一个“共显著物体识别平台”的前端项目。用户上传至少 2 张包含共同物体的图片后，前端会把图片提交到后端模型接口，并展示每张图片的原图、Mask 掩码图和叠加可视化图。

当前仓库已整理为 Cloudflare Workers 静态资源 + Worker 代理接口部署结构。模型权重、训练数据集、推理输出结果不放在前端仓库中，真实模型推理部署在独立后端服务器。

## 项目结构

```text
/
├─ public/
│  ├─ index.html
│  └─ assets/
├─ src/
│  └─ worker.js
├─ wrangler.toml
├─ README.md
└─ .gitignore
```

## 功能说明

- 初始状态不显示任何示例图片。
- 支持点击上传和拖拽上传图片。
- 至少需要上传 2 张图片，否则提示“请至少上传 2 张包含共同物体的图片”。
- 支持 JPG、PNG 格式，单张图片默认限制 20MB。
- 上传后显示本地图片预览，并支持移除和清空。
- 点击“开始体验”后，以 `multipart/form-data` 提交到同域接口 `/api/cosod`。
- 文件字段名为 `files`。
- 识别中按钮显示为“识别中...”并禁用。
- 识别完成后展示序号、原图、Mask 掩码、叠加效果和下载按钮。
- 兼容后端返回图片 URL，也预留 base64 图片返回格式。

## 前后端调用方式

浏览器访问 Cloudflare 前端：

```text
https://rcpr-identify.chenkang314.workers.dev/
```

前端会请求同域接口：

```text
POST https://rcpr-identify.chenkang314.workers.dev/api/cosod
```

Cloudflare Worker 会代理到华为云后端：

```text
POST http://120.46.136.60.sslip.io:8765/api/cosod
```

这样可以避免 HTTPS 前端直接请求 HTTP 后端时被浏览器拦截。

## 后端接口说明

请求格式：

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

## Cloudflare 部署方式

当前项目使用 Cloudflare Workers 静态资源部署，配置文件为 `wrangler.toml`。

Cloudflare 自动部署时使用：

```text
Deploy command: npx wrangler deploy
Root directory: /
```

`wrangler.toml` 中配置了：

```toml
main = "src/worker.js"

[assets]
directory = "./public"
binding = "ASSETS"
not_found_handling = "single-page-application"
run_worker_first = ["/api/*"]
```

其中 `/api/*` 会优先进入 Worker 代码，普通网页资源则从 `public/` 静态目录返回。

## 修改后端地址

后端地址在 `src/worker.js` 顶部配置：

```javascript
const BACKEND_ORIGIN = 'http://120.46.136.60.sslip.io:8765';
```

如果后端以后绑定了 HTTPS 域名，例如 `https://api.example.com`，只需要把这里改成：

```javascript
const BACKEND_ORIGIN = 'https://api.example.com';
```

然后提交 GitHub，Cloudflare 会重新部署。

## 注意事项

- 华为云安全组需要放行后端服务端口，例如 `8080`。
- 华为云服务器上需要启动模型后端，并提供 `/api/cosod`。
- Cloudflare Worker 只负责转发请求，不运行 Python 模型。
- 不要把模型权重提交到 GitHub，例如 `.pth`、`.pt`、`.ckpt`、`.onnx`。
- 不要把数据集提交到 GitHub，例如 `datasets/`、`dataset/`、`data/`。
- 不要把推理输出、上传缓存或结果目录提交到 GitHub，例如 `outputs/`、`results/`、`uploads/`、`predictions/`。
