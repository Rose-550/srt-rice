# 稻视界 · 水稻病虫害识别系统

> 桌面端（PyQt5 + YOLOv8n）识别 + 地块地图 + 防治建议 + 教学/科普 + 手机端联动的综合项目。
> 本文件是**目录地图**：先看这里，再决定文件放哪。详细整理方案见 [docs/工作区管理规范.md](docs/工作区管理规范.md)。

---

## 一、快速开始

```powershell
# 启动桌面端（唯一入口）
& 'D:\anoconda3\envs\rice-gpu\python.exe' rice_dp1_fluent.py

# 端到端自检（66+ 项）
& 'D:\anoconda3\envs\rice-gpu\python.exe' checks\fluent_smoke.py
```

- 环境：`D:\anoconda3\envs\rice-gpu`（Python 3.10，PyQt5 + ultralytics + torch）
- 部署权重：根目录 `best.pt`（= `runs/detect/yolov8n_clean/weights/best.pt` 的副本，见 [checks/deploy_swap/部署记录.md](checks/deploy_swap/部署记录.md)）
- 配置：根目录 `config.json`（含 API Key，**不要提交/外发**）、`settings.ini`

## 二、根目录 = 程序区（这些不要随便移动）

程序按“脚本所在目录”找资源（`BASE_DIR = Path(__file__).parent`），**以下文件/目录必须留在根目录**：

| 类别 | 文件/目录 | 说明 |
|---|---|---|
| 入口与模块 | `rice_dp1_fluent.py`、`rice_geo.py`、`rice_3d.py`、`edge_tts_lite.py`、`3d_viewer.html` | 主程序及其 import 依赖 |
| 模型 | `best.pt`、`yolov8n.pt`、`yolo26n.pt` | 部署权重 + 训练基座权重 |
| 资源 | `resources/`、`vendor/`、`手机端可跑版/` | 字体/图标/知识库/前端库/端侧 App |

> `icons/`、`formulas/`、`config/` 已归档（`archive/versions/`、`docs/images/formulas/`）；`maps/` 是程序启动时重写的运行时目录（见第三节）。
| 配置 | `config.json`、`settings.ini`、`schedule.json` | 运行时读写（config.json 已 gitignore） |
| 演示样本 | `手机端可跑版/samples/`、`datasets/dataset_fieldlike/images/val/`、`datasets/dataset_splitfix/images/val/` | 程序内“可用图片目录” |

## 三、运行时区（程序自己写，**不要手工往里放文件**）

| 目录 | 内容 |
|---|---|
| `sessions/` | 识别会话记录 |
| `inbox/`、`watch/` | 手机传图收件箱、中转状态、结果 CSV |
| `reports/` | App 生成的识别报告与缩略图 |
| `result/` | 旧版识别输出（历史，可归档） |
| `logs/` | App 运行日志、审计日志、训练日志（`logs/training/`） |
| `cache/`、`.cache/` | 天气/LLM 缓存 |
| `training/`、`teaching/`、`geo/`、`spray/`、`maps/`、`feedback/` | 各功能模块的本地数据 |

## 四、数据与训练

| 目录 | 内容 | 规矩 |
|---|---|---|
| `JPEGImages/`、`Annotations/`、`YOLOLabels/` | 原始图池（9296 张）+ VOC/YOLO 标注 | **只读**，不手改 |
| `datasets/dataset_*/` | 四套训练集构建（splitfix / dedup3 / fieldlike / fieldlike_ext） | 硬链接，**不能手删单张**，整目录重建；yaml 里的 `path` 为绝对路径 |
| `data/external/`、`data/_download/` | 外部数据集与下载缓存 | 只增不改；来源登记见 [docs/数据来源台账.md](docs/数据来源台账.md) |
| `runs/detect/` | 训练产物（权重、曲线、结果 CSV） | 基准实验保留；临时实验用完可归档 |
| `models/` | 非部署权重（`best_old.pt`、`best_names_fixed.pt`） | 不参与运行，归档用 |
| `checks/` | 自检脚本 + 证据（混淆矩阵、去重报告、部署记录等） | 每个结论都要能回溯到这里 |
| `cv_splits/`（若存在） | 5 折交叉验证划分 | 只读 |

- **现行对外基准**：簇级 5 折，mAP50 **0.9079 ± 0.0050**、mAP50-95 **0.5311 ± 0.0056**（YOLOv8n，见 [docs/实验记录.md](docs/实验记录.md) 实验十四）。
- 训练/数据脚本已放 `scripts/`（`download_rice_dataset.py`、`fix_pt_names.py`、`make_icons.py`、`get_asr_model.py` 等），用 `Path(__file__).parent.parent` 定位项目根。

## 五、文档与子项目

| 目录 | 内容 |
|---|---|
| `docs/` | 实验记录、数据台账、体检报告、使用手册、参赛材料、调研报告 |
| `docs/design/` | 设计规范与界面截图（原根目录 `设计规范.md` + `fluent_*.png`） |
| `docs/mobile/` | 手机端可行性评估、调研报告、路线图 |
| `docs/reference/` | 教程/参考资料（`yolo-for-beginners.html`） |
| `docs/reports/` | Word/PDF 正式报告（中期报告、详细报告等，按版本归档） |
| `docs/deploy/` | Nginx/HTTPS/中转/小程序部署笔记（①~⑨ 等） |
| `docs/images/` | 旧截图/对比图（`design/`）、视频（`media/`）、公式图（`formulas/`） |
| `docs/web/` | 可分享的演示/评估页面（model-eval、模型评估、rice-detection-progress） |
| `AI+教育大赛/` | 参赛材料子项目 |
| `科普/` | 科普内容子项目 |
| `云边联动方案C/` | 公网中转/小程序联调子项目 |
| `手机端可跑版/` | 浏览器端 ONNX 推理版（被主程序引用，勿移） |
| `archive/` | 旧版本脚本、旧日志、旧截图、历史打包产物、旧数据集布局 —— **只归档不删除** |

## 六、新文件放哪（约定）

| 你要放的东西 | 放这里 |
|---|---|
| 新功能代码 / 新模块 | 根目录（与主程序同级，按现有 import 方式） |
| 一次性脚本、数据处理、训练辅助 | `scripts/` |
| 新文档 / 报告 / 调研 | `docs/`（Word/PDF 进 `docs/reports/`） |
| 新数据集 | `data/external/<name>/`，先登记台账再动训练 |
| 实验截图 / 设计图 | `docs/images/` |
| 废弃版本 / 备份 / 旧产物 | `archive/YYYY-MM/` |
| 临时验证产物 | `reports/_scratch/<你的名字>/`（用完可删） |

## 七、重要文档索引

- [docs/实验记录.md](docs/实验记录.md) —— 所有实验、指标、基准与“不能碰的结论”
- [docs/数据来源台账.md](docs/数据来源台账.md) —— 每个数据集的来源/许可/遗留问题
- [docs/数据集体检报告.md](docs/数据集体检报告.md) —— 数据集结构与泄漏体检
- [docs/数据集补充方案_20261019.md](docs/数据集补充方案_20261019.md) —— 外部数据调研与补充计划
- [docs/使用手册_农户与学生工作流.md](docs/使用手册_农户与学生工作流.md) —— 面向使用者的操作说明
- [docs/工作区管理规范.md](docs/工作区管理规范.md) —— 本项目管理/整理规范
- [docs/Git使用指南.md](docs/Git使用指南.md) —— 手把手 Git 教程（初始化/日常五步/回滚/提交规范）

## 八、整理状态（2026-10-26）

- 已初始化 Git，基线 tag `baseline-20261026`；根目录从 175 个可见条目降到约 45 个。
- 已归档：历史版本、旧权重、训练日志、旧 `images/`+`labels/`+`dataset.yaml`、`build/`、`dist/`、`备份/`、`来时路/`、`reports/_scratch/`。
- 已归位：报告 → `docs/reports/`，部署笔记 → `docs/deploy/`，截图 → `docs/images/design/`，演示页 → `docs/web/`，设计规范 → `docs/design/`，手机端文档 → `docs/mobile/`，工具脚本 → `scripts/`，四套训练集 → `datasets/`。
- 待处理：大资源补齐口径、`大挑/`（149MB，未入库）、空目录清理。

## 九、克隆后需要补齐的资源

| 资源 | 获取方式 |
|---|---|
| `best.pt` | 从 `runs/detect/yolov8n_clean/weights/best.pt` 复制（见 `checks/deploy_swap/部署记录.md`） |
| `resources/fonts/` | 字体包（约 49MB，本地备份） |
| `resources/asr/` | `python scripts/get_asr_model.py` |
| `resources/tts/` | `python scripts/get_tts_model.py` |
| `resources/models/` | 3D 模型（约 32MB，本地备份） |
| `手机端可跑版/model.onnx` 等 | 端侧模型（本地备份） |
| `config.json` | 按自己的 API Key 新建，**不要提交** |
