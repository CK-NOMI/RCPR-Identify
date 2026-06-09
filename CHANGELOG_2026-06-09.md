# CHANGELOG 2026-06-09

## 16:36:57 推理阶段允许缺失 groundtruth

- 问题现象：`datasets/AQUA20_grouped/image/shark_2image` 仅包含图片，缺少对应 `datasets/AQUA20_grouped/groundtruth/shark_2image` 时，`dataset.py` 会在创建数据集或读取样本时失败。
- 诊断过程：检查 `CoData.__init__` 发现 groundtruth 目录校验不区分训练和推理；检查 `CoData.__getitem__` 发现每张图都会无条件打开同名 `.png` GT。
- 修复方案：训练模式继续强制要求 GT；推理模式不再要求 GT 目录或文件，缺失时创建同尺寸全零灰度 mask 作为占位，保持 `test.py` 的 batch 返回结构不变。
- 验证结果：已用缺失 `groundtruth/shark_2image` 的临时映射执行推理，模型成功处理 2 张图片，输出到 `predictions/ours_aqua20_shark_2image_u1_k20/CoCA/shark_2image`；Stage-1 输出到 `predictions/ours_aqua20_shark_2image_u1_k20/CoCA_stage1/shark_2image`。

## 17:10:31 前端封装多图共显著识别上传交互

- 问题现象：`前端/code.html` 只有静态页面结构，缺少文件上传、预览、提交 `/api/cosod`、加载态、结果渲染和错误提示逻辑，且原文件中文显示为乱码。
- 诊断过程：读取 `前端/code.html`，确认页面使用 TailwindCSS CDN 构建，已有顶部导航、上传卡、流程步骤、右侧优势卡、统计卡和结果空状态。
- 修复方案：重写 `前端/code.html` 为 UTF-8 中文完整页面；保留原有整体布局和视觉风格；新增原生 JS，支持多图选择/拖拽、至少 2 张校验、预览、清空、`multipart/form-data` 以 `files[]` 字段 POST 到 `/api/cosod`、加载状态、错误提示，以及原图/Mask/叠加效果结果渲染。
- 验证结果：通过本地静态服务打开 `http://127.0.0.1:8765/code.html`，确认标题、上传区、空结果区、初始无示例图片状态正常；点击“开始体验”在 0 张图片时显示“请至少上传 2 张图片后再开始识别。”；浏览器控制台无脚本错误，仅有 Tailwind CDN 生产环境提示。

## 17:23:51 接入 `/api/cosod` 后端服务

- 问题现象：前端已经会向 `/api/cosod` 提交 `files[]`，但项目中尚无后端接口接收上传图片、调用 SCoSPARC 推理并返回原图/Mask/叠加图 URL。
- 诊断过程：检查依赖文件发现项目未包含 Flask/FastAPI；检查 `test.py` 发现只支持固定 `datasets/CoCA` 等路径；检查 `utils.py` 发现 CRF 读图也固定读取 `./datasets/CoCA/image`。
- 修复方案：新增标准库后端 `cosod_backend.py`，同时服务 `前端/code.html`、`/api/health`、`/api/cosod` 和 `/outputs/...`；后端接收 multipart `files[]`，保存上传图片为临时 CoCA 兼容数据组，调用 `test.py` 推理，复制 Mask 并生成红色叠加图；给 `test.py` 增加 `--custom_img_path` 与 `--custom_gt_path`；给 `utils.py` 增加 `COSOD_CUSTOM_IMAGE_ROOT` 支持，使 CRF 可读取上传临时目录。
- 验证结果：`python -m py_compile cosod_backend.py test.py utils.py` 通过；使用 `D:\Anaconda\Anaconda\envs\scosparc\python.exe` 导入 `cosod_backend` 成功；后端已启动于 `http://127.0.0.1:8765/code.html`，`/api/health` 返回 `{"ok": true}`；浏览器确认页面标题、H1 与 `/api/cosod` 前端绑定正常。本次未主动提交图片触发模型推理。

## 17:44:28 修复前端识别完成后结果区域被隐藏

- 问题现象：用户上传 2 张图片并完成识别后，页面下方“识别结果”区域为空白，不显示原图、Mask 和叠加效果。
- 诊断过程：检查 `web_logs/20260609_173621_0c3976a2.log` 和 `web_outputs/20260609_173621_0c3976a2`，确认后端已经成功处理 2 张图片并生成 `original`、`mask`、`overlay` 文件；因此问题在前端渲染状态。检查 `setLoading(false)` 发现它在请求完成后仍会执行 `resultGrid.classList.add('hidden')`，把 `renderResults()` 刚渲染出的结果再次隐藏。
- 修复方案：调整 `前端/code.html` 的 `setLoading()`，只在 `isLoading=true` 时隐藏空状态和结果列表；识别结束调用 `setLoading(false)` 时不再改动结果列表显示状态。同时增加 `getCurrentFiles()`，提交时在内部数组为空时兜底读取 `fileInput.files`。
- 验证结果：确认后端真实输出已存在：`web_outputs/20260609_173621_0c3976a2/original`、`mask`、`overlay` 下各有 2 个文件；静态检查确认 `setLoading()` 中隐藏 `resultGrid` 的逻辑仅保留在 `isLoading` 分支内。

## 19:02:15 封装一键启动和停止脚本

- 问题现象：当前前后端功能已经可用，但下次使用仍需手动输入后端启动命令并打开浏览器。
- 诊断过程：确认后端服务入口为 `cosod_backend.py`，运行环境为 `D:\Anaconda\Anaconda\envs\scosparc\python.exe`，访问地址为 `http://127.0.0.1:8765/code.html`。
- 修复方案：新增 `start_cosod_web.ps1` / `start_cosod_web.bat`，用于检查健康接口、启动后端并打开页面；新增 `stop_cosod_web.ps1` / `stop_cosod_web.bat`，用于停止后端服务。
- 验证结果：执行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start_cosod_web.ps1` 成功，输出 `CoSOD web app is ready: http://127.0.0.1:8765/code.html`；未触发模型推理。

## 20:08:30 上传 GitHub 仓库并排除训练/预测数据

- 问题现象：需要将 `C:\COSOD\SCoSPARC-main课设版` 上传到 `CK-NOMI/RCPR-Identify`，但本地项目包含 `datasets/`、`predictions/` 等大目录；第一次全量提交约 6.48GB、152651 个文件，普通 Git 推送阶段因包体过大返回 HTTP 500。
- 诊断过程：扫描项目文件后确认最大单文件为 `models/dino_vitbase8_pretrain.pth`，大小约 343MB，超过 GitHub 普通文件限制，需使用 Git LFS；当前实际推理使用的业务 checkpoint 为 `checkpoints/baseline运行出的checkpoints/model_combo_base8-136_0.7291838924090067.pt`，约 7MB。根据用户补充，`datasets/` 和 `predictions/` 不需要上传。
- 修复方案：更新 `.gitignore` 排除 `datasets/`、`predictions/`、Web 临时输出和除当前 checkpoint 外的其它 `.pt/.pth` 权重；更新 `.gitattributes` 让 `models/dino_vitbase8_pretrain.pth` 走 Git LFS；更新 `README.md` 说明项目用途、启动方式、未上传数据目录和当前保留权重。使用临时克隆目录 `C:\COSOD\RCPR-Identify-upload` 重新暂存、提交并推送。
- 验证结果：推送成功，远端 `main` 从 `196c323bc` 更新到 `d4e107c4a0893bc5a758b5791bd6ad1e1ce19a26`；上传前检查确认暂存区没有 `datasets/` 和 `predictions/`；权重方面保留当前使用的 `model_combo_base8-136_0.7291838924090067.pt`，以及由 LFS 管理的 `models/dino_vitbase8_pretrain.pth`。
