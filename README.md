# 稻视界 · 水稻病虫害识别系统

> 桌面端 PyQt5 + YOLOv8n 识别 + 地块地图 + 防治建议 + 教学科普 + 手机端联动的代码仓库。
> **本仓库只包含程序代码及其运行依赖**；数据集、模型权重、日志、配置密钥、比赛材料和文档报告保留在本地工作区，不入库。

## 克隆后先做什么

1. 安装 **Python 3.10**。
2. 安装 PyTorch（二选一）：

   ```powershell
   # GPU（CUDA 12.1）
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

   # 或者 CPU
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
   ```

3. 安装其余核心依赖：

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. 把 **`best.pt`** 放到项目根目录。**没有它程序无法启动**，会在创建主窗口时报“未找到模型权重文件”。
   - 向项目维护者获取；或从训练产物 `runs/detect/yolov8n_clean/weights/best.pt` 复制。
5. 启动桌面端：

   ```powershell
   python rice_dp1_fluent.py
   ```

6. 跑一遍离屏自检：

   ```powershell
   python checks\fluent_smoke.py
   ```

   自检产物写到 `checks/_fluent_out/`，不会污染运行时目录。
7. 需要语音识别、本地语音合成或 Word 报告时，再安装可选依赖：

   ```powershell
   python -m pip install -r requirements-optional.txt
   ```

## 运行环境

- Python 3.10
- 核心依赖见 `requirements.txt`；可选依赖见 `requirements-optional.txt`
- GPU 版 PyTorch 需要与本机 CUDA 匹配；没有 GPU 时程序会退回 CPU
- `config.json` 不入库；程序首次启动用默认配置，AI 分析需要在设置页填 DeepSeek / DashScope Key，程序会写回本地 `config.json`
- `settings.ini`、`schedule.json` 由程序自动创建

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

## 模型与资源

以下内容**不在仓库里**，需要本地补齐：

| 资源 | 获取方式 |
|---|---|
| `best.pt` | 向维护者获取，或从 `runs/detect/yolov8n_clean/weights/best.pt` 复制；**缺失无法启动** |
| `datasets/dataset_fieldlike/`、`datasets/dataset_splitfix/` | 训练集/评测集；缺失时教学题库只能用 `手机端可跑版/samples/` 的少量图片 |
| `data/raw/` | 原始图池与标注（只读） |
| `resources/fonts/` | 字体包（缺失回退系统字体） |
| `resources/asr/` | `python scripts/get_asr_model.py` |
| `resources/tts/` | `python scripts/get_tts_model.py` |
| `resources/models/` | 3D 模型（缺失时 3D 页面没有模型） |
| `手机端可跑版/model.onnx`、`*.wasm` | 手机端网页版模型（缺失时手机端不能推理） |

`vendor/` 里的 `pypdf`、`qrcode` 是随程序分发的小型依赖，不需要单独 pip 安装。

## 团队协作

- 开始前先 `git pull`；改动前确认工作区干净。
- 改完程序跑一遍 `checks/fluent_smoke.py`，不要把失败项说成“已完成”。
- 提交前 `git status` 确认没有误加大文件或密钥。
- 以下内容已被 `.gitignore` 排除，**不要提交**：`best.pt`、`datasets/`、`data/`、`logs/`、`sessions/`、`config.json`、`settings.ini`、`docs/`、`AI+教育大赛/`、`科普/`、`大挑/`。
- 文档、比赛材料和截图放在本地工作区即可，仓库只保留代码。
- 训练集是硬链接：**不要按目录大小判断占用，也不要手删单张图片**。
- `datasets/*/dataset.yaml` 的 `path:` 是绝对路径，迁移后需同步修改。
