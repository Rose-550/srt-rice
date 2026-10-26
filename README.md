# 稻视界 · 水稻病虫害识别系统

> 桌面端 PyQt5 + YOLOv8n 识别 + 地块地图 + 防治建议 + 教学科普 + 手机端联动的代码仓库。
> **本仓库只包含程序代码及其运行依赖**；数据集、模型权重、日志、配置密钥、比赛材料和文档报告保留在本地工作区，不入库。

## 运行环境

- Python 3.10
- PyQt5、ultralytics、torch（GPU `device=0`）、opencv-python、Pillow、requests、qfluentwidgets、markdown
- 模型权重 `best.pt` 放在项目根目录

## 启动

```powershell
& 'D:\anoconda3\envs\rice-gpu\python.exe' rice_dp1_fluent.py
```

## 自检

```powershell
& 'D:\anoconda3\envs\rice-gpu\python.exe' checks\fluent_smoke.py
```

自检产物写到 `checks/_fluent_out/`，不会污染运行时目录。

## 代码结构

| 路径 | 说明 |
|---|---|
| `rice_dp1_fluent.py` | 唯一程序入口（PyQt5 + YOLOv8n） |
| `rice_geo.py` | 地块地图 / 地理数据 |
| `rice_3d.py`、`3d_viewer.html` | 3D 场景 |
| `edge_tts_lite.py` | 离线语音合成 |
| `resources/` | 字体、图标、品牌图、药剂图、语音/3D 模型（大文件不入库） |
| `vendor/` | leaflet、pypdf、qrcode |
| `手机端可跑版/` | 浏览器端 ONNX 推理版 |
| `云边联动方案C/` | 手机传图中转 / 小程序联调 |
| `checks/` | 离屏自检与审计脚本 |
| `scripts/` | 训练 / 数据 / 模型工具脚本 |

## 不入库的内容

以下内容保留在本地工作区，`.gitignore` 已排除：

- `datasets/`、`data/`：训练集、原始图池、外部数据
- `best.pt`、`yolov8n.pt`、`yolo26n.pt`：模型权重
- `resources/fonts/`、`resources/asr/`、`resources/tts/`、`resources/models/`：大体积运行资源
- `logs/`、`sessions/`、`inbox/`、`reports/` 等运行时目录
- `config.json`、`settings.ini`：本地配置（含 API Key，**绝不入库**）
- `docs/`、`AI+教育大赛/`、`科普/`、`大挑/`：文档、比赛与科普材料
- `archive/`：历史归档

## 大资源补齐

| 资源 | 获取方式 |
|---|---|
| `best.pt` | 从 `runs/detect/yolov8n_clean/weights/best.pt` 复制 |
| `resources/asr/` | `python scripts/get_asr_model.py` |
| `resources/tts/` | `python scripts/get_tts_model.py` |
| `resources/fonts/`、`resources/models/` | 本地备份 |
| `手机端可跑版/model.onnx` 等 | 本地备份 |

## 说明

- `CLAUDE.md` 是本地工作区的开发说明，保留在仓库里方便协作。
- 训练集与原始图池是硬链接，**不要按目录大小判断占用，也不要手删单张图片**。
- `datasets/*/dataset.yaml` 的 `path:` 是绝对路径，迁移后需同步修改。
