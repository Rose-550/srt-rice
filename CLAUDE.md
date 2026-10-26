# CLAUDE.md — 稻视界（rice）

桌面端水稻病虫害识别系统：PyQt5 界面 + YOLOv8n 检测 + 地块地图/防治建议 + 教学科普 + 手机端联动。
**目录地图与文件归属先看 [README.md](README.md)**；做整理/归档类改动先看 [docs/工作区管理规范.md](docs/工作区管理规范.md)。

## 环境与常用命令

- Python 环境：`D:\anoconda3\envs\rice-gpu\python.exe`（3.10，PyQt5 + ultralytics + torch，GPU 为 device=0）
- 唯一程序入口：`rice_dp1_fluent.py`（约 1.3 万行，`BASE_DIR = Path(__file__).parent`）
- 依赖模块：`rice_geo.py`、`rice_3d.py`、`edge_tts_lite.py`、`3d_viewer.html`

```powershell
# 启动桌面端（会弹 GUI，只在用户要求时运行）
& 'D:\anoconda3\envs\rice-gpu\python.exe' rice_dp1_fluent.py

# 端到端自检（离屏，不需要人点界面；改完主程序应跑一遍）
& 'D:\anoconda3\envs\rice-gpu\python.exe' checks\fluent_smoke.py
```

自检产物写到 `checks/_fluent_out/`（已隔离日志/报告/反馈），不会污染运行时目录。另有 `checks/app_regression.py`、`checks/geo_smoke.py`。

## 硬约束（改代码前必读）

- **根目录相对路径写死**：主程序按 `BASE_DIR` 找资源。以下必须留在根目录，移动即功能失效：
  `rice_dp1_fluent.py`、`rice_geo.py`、`rice_3d.py`、`edge_tts_lite.py`、`3d_viewer.html`、
  `best.pt`、`resources/`、`vendor/`、`icons/`、`formulas/`、`config/`、`maps/`、`手机端可跑版/`、
  `手机端可跑版/samples/`、`dataset_fieldlike/images/val/`、`dataset_splitfix/images/val/`。
- **根目录的版本副本是历史，不要当入口**：`rice_dp1*.py` 的其他副本、`*.diff`、`main.py`、`1.py`、`rice_1.py`、
  `deep learning.py` 都不是当前程序。改动只改 `rice_dp1_fluent.py`，**不要再生成 `xxx - 副本 (2).py` 这类文件**。
- **运行时目录由程序自己写，不要手工放文件**：`sessions/`、`inbox/`、`watch/`、`reports/`、`logs/`、
  `cache/`、`result/`、`training/`、`teaching/`、`geo/`、`spray/`、`maps/`。
- 新增一次性脚本放 `scripts/`（目前尚未创建，需要时再建）；新文档放 `docs/`；临时验证产物放
  `reports/_scratch/`；废弃文件只归档到 `archive/YYYY-MM/`，**不删除**。

## 数据规则

- `JPEGImages/`、`Annotations/`、`YOLOLabels/` 是原始图池（9296 张），**只读**，不手改、不重排。
- `dataset_splitfix/`、`dataset_dedup3/`、`dataset_fieldlike/`、`dataset_fieldlike_ext/` 四套训练集：
  - 图片之间是**硬链接**，看起来各占 500MB 实际只有一份 —— **不要按目录大小决定删除**；
  - **不能删单张图片**，要改就整目录重建；
  - 各自 `dataset.yaml` 里 `path:` 是**绝对路径**，迁移/重建后必须同步修改。
- `data/external/` 新数据集先进 [docs/数据来源台账.md](docs/数据来源台账.md) 登记来源/许可，再接入训练。

## 训练与评估

```powershell
# 训练模板（参数在 docs/实验记录.md 中逐字记录，改基准实验前先对照）
& 'D:\anoconda3\envs\rice-gpu\Scripts\yolo.exe' detect train model='C:\Users\罗浔\Desktop\rice\yolov8n.pt' data='C:\Users\罗浔\Desktop\rice\dataset_fieldlike\dataset.yaml' epochs=50 imgsz=640 batch=16 device=0 seed=0 deterministic=True close_mosaic=10 workers=4 exist_ok=True project='C:\Users\罗浔\Desktop\rice\runs\detect' name=<实验名>

# 验证
& 'D:\anoconda3\envs\rice-gpu\Scripts\yolo.exe' detect val model='...\weights\best.pt' data='...\dataset.yaml' split=val imgsz=640 batch=16 device=0
```

- **现行对外基准**（簇级 5 折，YOLOv8n）：mAP50 **0.9079 ± 0.0050**，mAP50-95 **0.5311 ± 0.0056**。
  出处与溯源见 [docs/实验记录.md](docs/实验记录.md) 实验十四。**不要凭记忆复述指标，引用前回该文件核对。**
- 训练耗时长（单次约 0.7–1 小时），**未经用户明确要求不要启动训练**；启动前确认没有别的进程占 GPU。
- 产物进 `runs/detect/<name>/`；根目录 `best.pt` 是部署权重，由 `runs/detect/yolov8n_clean/weights/best.pt`
  复制而来（见 `checks/deploy_swap/部署记录.md`），**换权重需同步更新该记录**。
- 训练日志编码不一致：`train_yolov8n_clean.console.log` 是 **UTF-16LE**，其余是 UTF-8。用脚本读之前先转码，
  否则中文用户名会被吞成 `C:\Users\\Desktop\...`。这些日志是数字溯源的证据，**不要删**。

## 安全

- `config.json` 含**明文 API Key**（DeepSeek / DashScope）与 relay token，已被 `.gitignore` 排除。
  **不要把内容打印到对话、复制到别处、写进文档或提交到任何仓库。**
- 项目已初始化 git 仓库（`main` 分支，基线 tag `baseline-20261026`）。批量移动/回滚前先 `git status`；`config.json`、数据集、权重、日志和大文件已由 `.gitignore` 排除。

## 检索与性能提示

项目根目录大且杂，搜索时排除这些目录，否则又慢又噪：
`dataset_*/`、`JPEGImages/`、`Annotations/`、`YOLOLabels/`、`runs/`、`node_modules/`、`.mobile_probe/`、
`vendor/`、`build/`、`dist/`、`备份/`、`archive/`、`*.console.log`（单文件最大 10MB）。
定位代码时优先用函数名/字符串 grep `rice_dp1_fluent.py`，它很大，不要整文件通读。

## 改动习惯

- 主程序是单文件大程序：**外科式修改**，按函数/类定位，保持现有命名与中文注释风格。
- 改完后跑 `checks\fluent_smoke.py`，把失败项和输出如实报告，不要只说"已完成"。
- 涉及界面/截图时用离屏模式（`QT_QPA_PLATFORM=offscreen`），不要主动弹 GUI 打断用户。
