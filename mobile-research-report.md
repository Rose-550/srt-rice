# 「稻视界」手机端落地技术调研报告

- 调研人：mobile-stack-research（只读调研任务 task-1）
- 调研日期：2026-10-12（北京时间）
- 工作目录（只读）：`C:\Users\罗浔\Desktop\rice`
- 本文唯一写入文件：`mobile-research-report.md`
- 现有端形态（据 Lead 提供 + 本地 config.json 核对）：PyQt5 桌面端 `rice_dp1_optimized.py`，依赖 PyQt5 / cv2 / torch / ultralytics / markdown / requests；权重 `best.pt`、`yolov8n.pt`、`yolo26n.pt`；`config.json` 配置 `deepseek-chat` + `qwen-vl-max`。**注意：config.json 内含明文 API Key，迁移到移动端时必须改为服务端代理或系统密钥库，禁止把 Key 打进 APK/IPA。**

## 0. 证据等级与阅读约定

| 标记 | 含义 |
| --- | --- |
| 【实测·官方】 | 厂商/官方文档或官方模型卡中给出的实测数字，已实际抓取页面 |
| 【实测·二手】 | 技术博客/第三方整理的实测数字，已抓取页面但非一手厂商 |
| 【官方文档】 | 官方文档中对能力/限制的描述（非性能数字） |
| 【推断】 | 由多组已知数据外推得出，**不是实测**，用于量级判断 |
| 【不确定】 | 未能找到可靠来源，或来源过旧/未抓取到正文，需要进一步验证 |

---

## 1. 结论速览（TL;DR）

1. **手机端跑 YOLO 已完全成熟**，但"必须换运行时、换语言、换推理格式"。PyTorch/ultralytics 无法在 Android/iOS 上原样运行。
   - 旗舰机 NPU 上 YOLOv8-n 640×640 纯推理 **1–4 ms**；中端机（骁龙 7 系级）**2–10 ms**；CPU(XNNPACK) 路线旗舰机端到端 **36–53 ms**。
2. **推荐链路**：`.pt → ONNX（simplify, opset 12+，`nms=None`）→ LiteRT(.tflite)/NCNN/QNN`；把 NMS 与 letterbox 预处理放在 Kotlin/Dart 侧实现（除 CoreML 外几乎都不能把 NMS 图内导出）。
3. **"视觉大模型诊断"不要放端侧**：可用的最小可用 VLM（MiniCPM-V 4.0，4.1B）在 iPhone 16 Pro Max 上是 **首 token < 2 s、>17 tok/s**【实测·二手】；但 8.1B 的 MiniCPM-V 2.6 权重就有 **16.2 GB（BF16）**【实测·官方】，手机放不下。**走云端 API 是唯一性价比合理的路线**，单次诊断成本约 **0.0008 元**（按 qwen-vl-plus 快照价估算）。
4. **UI 框架：明确结论 —— Chaquopy 和 Kivy/python-for-android 都【不能】直接复用 PyQt5 代码。**
   - Chaquopy 的 Android wheel 仓库里**没有 PyQt5/PySide**，`torch` wheel 最后更新停在 **2021-05-22**；且 Chaquopy **不支持 iOS**。
   - p4a/Kivy 只有 SDL2 bootstrap，**没有 PyQt5 recipe**，UI 必须用 Kivy 重写。
   - 结论：PyQt5 UI 层 100% 重写；纯 Python 业务逻辑（规则库、prompt 拼接、结果格式化）可以移植，但必须做"对拍测试"。
5. **上架与合规是日历时间的主要风险**：App 备案管局终审 **20 个工作日**【实测·官方】；软著 **受理后 30–60 个工作日**【二手】；大模型/AI 相关备案实践 **3–4 个月、甚至 4–10 个月**【二手】。这些是**等待时间，不能靠加班压缩**。
6. **三档工期（1 名熟练移动开发者，人日）**：
   - **A 轻量 MVP：12–20 人日**（日历 3–5 周，含审核）
   - **B 完整离线 App：35–60 人日**（日历 2–3 个月）
   - **C 完整 App + GIS + 用户体系 + 知识库：70–130 人日**（日历 3.5–5 个月，不含备案等待）
   - 若需 iOS 双端，另加 **+60%~+100%**（Chaquopy 不支持 iOS，端侧推理要再写一遍 CoreML/Swift）。

---

## 2. 手机端 YOLO 目标检测方案对比

### 2.1 横向对比总表（YOLOv8 / YOLO11 系列）

| 方案 | 对 YOLOv8/v11 支持成熟度 | 是否需要自定义算子 | 量化后精度损失量级 | 中端安卓 640×640 单帧耗时量级 | 关键证据 |
| --- | --- | --- | --- | --- | --- |
| **LiteRT / TFLite**（Google） | 高。Ultralytics 官方支持 `format=litert`，v8/v11 全任务（detect/segment/classify/pose/obb）✅【官方文档】 | 通常**不需要**；但 NMS 需在宿主侧实现，GPU 委托默认 FP16 | w8a32（动态）"minimal accuracy loss"（官方措辞，无数值）；w8a16 更高精度 | 旗舰 8 Elite Gen 5：CPU **52.2 ms**、GPU **15.8 ms**（端到端含前后处理）；Pixel 10：CPU 53.3 / GPU 45.5；S26：CPU 36.7 / GPU 16.4；天玑9500：CPU 45.4 / GPU 26.6【实测·官方】。中端外推：**CPU 60–150 ms，GPU 25–60 ms**【推断】 | docs.ultralytics.com/integrations/litert.md |
| **NCNN**（腾讯） | 高。Ultralytics 官方支持 `format=ncnn`（v8/v11 detect/segment/classify/pose/obb）【官方文档】 | 通常不需要；NMS 需 C++/Kotlin 侧实现；Vulkan 后端是移动端独有 | INT8 用 ncnn2table(KL/ACIQ)+ncnn2int8；PTQ 无 QAT，**经验损失 1–3 个 mAP50 百分点**【不确定，见 §2.3】 | 骁龙 865（≈今日中端）YOLOv8n 640 **60–70 FPS ≈ 14–17 ms**（CPU/多核，博客实测）【实测·二手】。中端 7 系/天玑 8000 外推 **15–40 ms**【推断】 | blog.csdn.net/ZhuBin365/article/details/157223786 |
| **MNN**（阿里） | 高。Ultralytics 官方支持 `format=mnn`，v8/v11/v26 七类任务【官方文档】；淘宝/天猫/钉钉等 30+ App 在用 | 不需要；量化集成进 MNNConvert，支持混合精度/敏感度分析 | PTQ + QAT 均支持；KL + 学习步长校准；INT4 实验性 | 与 NCNN 同量级，**15–40 ms**【推断】（未找到官方中端机对照实测，**不确定**） | docs.ultralytics.com/integrations/mnn.md；estelledc.github.io（MNN vs ncnn 对照） |
| **ONNX Runtime Mobile** | 中高。可直接吃 ONNX；Android 有 QNN EP / NNAPI EP，官方有 "Object detection … with YOLOv8" 教程【官方文档】 | 不需要自定义算子即可跑；但 **NMS 不在图内**，官方教程里自己做后处理；端到端（含 NMS）图导出受限 | INT8 需 ORT 量化工具 + 校准集 | 走 QNN EP 时与 Qualcomm AI Hub 数字同量级：骁龙 7 Gen 4 **w8a8 1.905 ms / w8a16 6.458 ms**（NPU，纯推理）【实测·官方】 | onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider；hf-mirror.com/qualcomm/YOLOv8-Detection |
| **Paddle Lite**（百度） | **偏低**。Ultralytics 无 Paddle Lite 导出集成；YOLOv8 转换时 **SiLU/Swish 算子不支持**，需替换激活或自己加算子【实测·二手】；仓库维护活跃度存疑【不确定】 | **是**（SiLU/Swish 缺失） | INT8 支持，但算子替换后精度需重新验证 | 未找到可靠中端机 YOLOv8 实测数字【不确定】 | blog.gitcode.com/da161a03187313f00a3f5675387a440c.html |
| **ExecuTorch**（Meta/PyTorch） | 中。已发布 1.0（Meta 2026 年发布，新闻标题可查，正文未抓取）【不确定】；Ultralytics 支持 `format=executorch`，但**目前只导出 FP32（XNNPACK 后端），不带量化**【官方文档】 | 不需要；NMS 宿主侧实现 | 官方导出链路暂无量化 → 无量化精度损失，但也没有量化收益 | 未找到中端机 YOLO 实测数字【不确定】；XNNPACK 属 CPU 路线，参照 LiteRT CPU 数字 36–53 ms（旗舰），中端更高 | docs.ultralytics.com/integrations/executorch.md |
| **Qualcomm QNN**（骁龙 NPU） | 高（仅骁龙）。Ultralytics 官方 `format=qnn`；AI Hub 有现成 YOLOv8-Detection 资产 | 不需要；w8a8 由 AI Hub Workbench 编译 | w8a16 精度更高、w8a8 更快；**同一芯片 w8a8 1.013 ms vs w8a16 2.916 ms（8 Gen 3）**，说明 w8a8 有明显速度换精度权衡【实测·官方】 | 骁龙 8 Gen 3 NPU：float 3.617 / w8a16 2.916 / w8a8 **1.013 ms**；骁龙 7 Gen 4：w8a16 6.458 ms（ONNX-RT）、QNN_DLC w8a16 4.027 ms、w8a8 1.905 ms；QCS6490（7 系级）：w8a8 4.709 ms【实测·官方】（**均不含前后处理与 NMS**） | hf-mirror.com/qualcomm/YOLOv8-Detection |
| **Core ML（ANE）**（Apple） | 高（仅 iOS）。Ultralytics 官方 `format=coreml`，v8/v11 全任务；官方 iOS SDK + Flutter 插件直接跑 ANE | 不需要；**唯一支持图内嵌 NMS 的格式**（detect/segment/pose，需静态 shape）【官方文档】 | INT8 CFG 量化；官方给出 INT8 资产实测 | iPhone 17 Pro：YOLO26n detect 640 **3.2 ms**（CPU+ANE 端到端）/ 9.2 ms（CPU only）【实测·官方】 | docs.ultralytics.com/integrations/coreml.md |

### 2.2 关键口径提醒（避免被"数字"误导）

- **Qualcomm AI Hub 的数字（1–10 ms）不含预处理 / NMS / 后处理**，是纯网络前向。真实 App 里端到端要乘 2–5 倍（见 Ultralytics 端到端实测 15.8 ms GPU / 52.2 ms CPU）。
- Ultralytics 的 LiteRT/QNN 端到端实测用的是 **YOLO26n**（2026 年的新头，NMS-free）；YOLOv8n 数字会略高（多一个 NMS 步骤）。
- 上表"中端安卓"口径：**骁龙 7 系 / 天玑 8000 系**。直接实测数据只有 Qualcomm AI Hub 的 **骁龙 7 Gen 4 / QCS6490**（NPU）；CPU/GPU 路线的中端数字是【推断】。
- 骁龙 865 那条 ncnn 60–70 FPS 的博客属于 AI 生成/整理性质，**仅作量级参考**。

### 2.2.1 与项目自有模型的对应关系（口径校准）

Lead 侧实测：\`best.pt\` **3.012M 参数、.pt 5.95 MB**，ONNX 导出 **opset 19 / 11.68 MB**，算子全部为标准算子（Conv/Mul/Sigmoid/Concat/Resize/MaxPool），**无自定义算子**。
→ 该规模与 **YOLOv8-n（3.18M 参数 / float 12.2 MB）** 同级，因此本报告引用的 yolov8n 端侧耗时与体积数据**可以直接套用**，不需要按大模型折算。
→ "无自定义算子"这一条非常关键：它意味着 **LiteRT / NCNN / MNN / QNN / CoreML 五条链路都可以直接转换，不需要写自定义算子**（Paddle Lite 的 SiLU 坑对本模型不适用）。

### 2.3 量化精度损失（这一项来源最弱，明确标注）

- **官方只给定性描述**：Ultralytics 导出文档对 INT8 的措辞是 "minimal accuracy loss"，**没有给 mAP 数字**【官方文档】。
- **可核实的结构性事实**：Qualcomm 同一芯片同一模型，w8a8 比 w8a16 快 2–3 倍（8 Gen 3：1.013 ms vs 2.916 ms），说明 **w8a8 的"精度余量"被换成了速度**。工程上稳妥做法：**先上 w8a16，w8a8 只在精度验收通过后启用。**
- **行业经验值（无一手来源，标【不确定】）**：YOLOv8-n 级模型 INT8 PTQ 后 mAP50 通常下降 **0.5–3 个百分点**；分类头/小目标层最敏感，需要 per-channel 量化或该层保留 FP16。
- **本项目具体风险**：`best.pt` 是**自训练的病害检测模型（类别少、病害斑点多尺度）**，量化对**小斑点召回**的影响必须用你自己的验证集实测，不能引用 COCO 经验值。这一项需要在里程碑里单独排 **2–4 人日**做精度验收。

---

## 3. 端侧模型格式转换链路（.pt → ONNX → 各端侧格式）

### 3.1 推荐链路与工具

```
best.pt (PyTorch, ultralytics)
  └─ [1] yolo export format=onnx imgsz=640 simplify=True opset=12  → best.onnx   (raw output, nms=None)
        └─ [2a] LiteRT : 需 Linux/macOS 环境 → best.tflite (quantize='w8a32' 或 8 + data 校准)
        └─ [2b] NCNN   : pnnx best.pt  或  onnx2ncnn best.onnx  → .param/.bin
                         再 ncnn2table + ncnn2int8 → INT8（KL 校准）
        └─ [2c] MNN    : MNNConvert -f ONNX --weightQuantBits 8 → .mnn
        └─ [2d] QNN    : Qualcomm AI Hub / QAIRT 编译 → .dlc（w8a16 / w8a8）
        └─ [2e] CoreML : yolo export format=coreml nms=True（detect/segment/pose, 静态 shape）→ .mlpackage
  └─ [3] 端侧后处理：letterbox 还原 + 置信度过滤 + NMS（Kotlin/Dart C++ 侧实现）
```

### 3.2 各环节的坑（按踩坑概率排序）

| # | 坑 | 具体表现 | 规避 |
| --- | --- | --- | --- |
| 1 | **NMS 无法导出到图里**（除 CoreML） | Ultralytics `nms` 参数语义：`None`=输出 raw one-to-many 预测、外部做 NMS；`True`=在支持的格式内嵌 NMS；`False`=改用 NMS-free 头（YOLO26/YOLOv10）。**CoreML 内嵌 NMS 仅支持 detect/segment/pose 且必须静态 shape**【官方文档】 | 端侧把 NMS 写成 Kotlin/C++；或换 NMS-free 头模型；不要在 TFLite/NCNN 上指望图内 NMS |
| 2 | **LiteRT 导出只能在 Linux x86_64 / macOS 上做** | "LiteRT export is currently supported on Linux x86_64 and macOS"【官方文档】。你现在是 Windows 开发机 → **必须上 WSL2 或 CI** | 用 WSL2/Ubuntu Docker 做导出，产物提交到仓库 |
| 3 | **INT8 静态量化必须给校准数据** | `quantize=8` 需要 `data=`（如 `coco8.yaml`）或 `fraction`；校准集分布与实际拍照场景不匹配 → 精度暴跌【官方文档 + 二手】 | 校准集**必须用你自己的田间拍摄图**（100–500 张），不要用 COCO |
| 4 | **dynamic shape** | `dynamic=True` 只对 TorchScript/ONNX/OpenVINO/TensorRT/**CoreML/MNN** 有效；**LiteRT/NCNN/QNN 走静态** | 固定 640×640，端侧自己 resize；宽高比靠 letterbox 补齐 |
| 5 | **预处理不一致** | TFLite 传统链路是 NHWC + `Identity` 输出名；新的 LiteRT 直追 PyTorch 图产生 **NCHW + float 输入**，签名名不同。照抄旧代码会"跑通但框全错"【官方文档】 | 读 tensor shape/signature，不要硬编码；用同一张图在 PC 与端侧对拍 |
| 6 | **算子缺失** | Paddle Lite 对 YOLOv8 的 SiLU/Swish 不支持；ncnn 有 PNNX 静默失败（denormalized float 常量）案例 | 选 NCNN/MNN/LiteRT 这类已被 Ultralytics 官方覆盖的链路；转换后做逐层输出对拍 |
| 7 | **量化工具链割裂** | NCNN 用独立的 `ncnn2table`+`ncnn2int8`；MNN 集成在 `MNNConvert`；ORT 用 `onnxruntime.quantization` | 把量化脚本版本化，记录每个模型的校准集 hash |
| 8 | **模型体积** | YOLOv8n float 12.2 MB → w8a16 3.60 MB → w8a8 3.25 MB【实测·官方】（Qualcomm 模型卡） | APK 里带模型没问题；但 Chaquopy 每个 ABI 再加数 MB |

---

## 4. 手机端"视觉大模型诊断"：两条路线

### 4.1 路线 A：端侧小 VLM

| 模型 | 参数量 | 显存/内存占用 | 速度 | 手机可行性 | 来源 |
| --- | --- | --- | --- | --- | --- |
| **MiniCPM-V 2.6** | **8.1B**（HF safetensors 实测总数 8,099,175,152） | BF16 权重 **16.2 GB** | — | **不可行**（4bit 仍需 ≈4 GB 常驻 + KV cache） | hf-mirror.com/openbmb/MiniCPM-V-2_6 |
| **MiniCPM-V 4.0** | 4.1B | Q4 级 ≈2.5–3 GB【推断】 | iPhone 16 Pro Max：**首 token < 2 s，解码 > 17 tok/s**，无过热 | **可行但吃紧**（8 GB RAM 机型边缘；需 Q4 + 限制图像 token） | blog.gitcode.com/fe0c1a7d…（整理官方模型卡） |
| **MiniCPM-V 4.6** | 1.3B（SigLIP2-400M + Qwen3.5-0.8B） | Q4 级 ≈1 GB【推断】 | 未取到实测 | **最可行**（但能力显著下降）【不确定】 | gitcode 相关文章标题（未抓正文） |
| **SmolVLM-Instruct** | **2.246B**（HF 实测） | BF16 文件 **4.49 GB** → INT8 ≈2.3 GB【推断】 | 未取到手机实测 | 中低端机吃紧；256M/500M 版本更适合，但其"<1 GB GPU RAM"说法**未验证** | hf-mirror.com/HuggingFaceTB/SmolVLM-Instruct |
| **Qwen2.5-VL-3B** | 3B 级（3.75B） | BF16 ≈7.5 GB【推断】 | OpenCompass 均分 64.5（低于 MiniCPM-V 4.0 的 69.0） | **不可行**（4bit ≈2.2 GB 勉强，但图像 token 一多就爆） | hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct；gitcode 评测整理 |

**结论（路线 A）**：端侧 VLM 在 2026 年是"能跑、但不可用于生产诊断"。
- 内存：中端安卓 8 GB RAM，系统+相机+YOLO 已占 2–3 GB，留给 VLM 的预算 ≈3–4 GB。
- 延迟：首 token <2 s 只是**旗舰 iPhone** 的数字；中端安卓 CPU 推理预计 **5–20 s 首 token**【推断，**不确定**】。
- 精度：4B 级 VLM 在专业病害诊断上**明显弱于** qwen-vl-max / DeepSeek 云端大模型，而本项目诊断结论直接面向农户，错判代价高。
- **建议**：端侧 VLM 只作为"离线降级方案 + 只在高端机开启"的可选项，不进 MVP 主线。

### 4.2 路线 B：云端 API 调用（推荐）

| 项目 | 数值 | 来源 |
| --- | --- | --- |
| DeepSeek 当前模型 | `deepseek-flash`（**支持 Vision**，1M 上下文，384K 最大输出）；`deepseek-v4-pro`（**不支持 Vision**）；旧名 `deepseek-chat`/`deepseek-v4-flash-vision-exp` 已被退役并按 Flash 计费 | api-docs.deepseek.com/quick_start/pricing/（2026-10-12 抓取） |
| DeepSeek 价格 | 输入（缓存未命中）**$0.15/百万 token**（非高峰）/ $0.3（高峰）；输出 **$0.6/百万**（非高峰）/ $1.2（高峰）。高峰=UTC 周一至周五 01:00–04:00 与 06:00–10:00 | 同上 |
| 阿里云百炼 Qwen-VL | `qwen-vl-plus-2025-07-10`：输入 **0.00015 元/千 token**（=0.15 元/百万）、输出 **0.0015 元/千 token**（=1.5 元/百万）；上下文 32,768，单图最大 16,384 token；限流 **60 QPM / 100,000 TPM** | help.aliyun.com/zh/model-studio/vision-white |
| 单次诊断成本估算 | 一张 1280×720 图 ≈1,175 图像 token + ~200 文本 token；输出 ~400 token → **≈0.0008 元/次**（qwen-vl-plus 快照价） | 由上面单价推导【推断】 |
| 用量成本 | 1,000 次/天 ≈ **0.8 元/天 ≈ 24 元/月**；1 万次/天 ≈ 240 元/月 | 同上【推断】 |
| `qwen-vl-max`（现有 config） | **未取到官方现价**【不确定】。百炼当前主推 qwen3.x 系列；`qwen-vl-max` 属旧型号，价格与可用性需在控制台确认 | — |

**实现复杂度**：低。现状已经是 HTTP 调 DashScope/DeepSeek，手机端只是把 `requests` 换成 `OkHttp`/`Dio`，**1–2 人日**即可打通（不含图片压缩、重试、流式）。

**必须做的工程加固（容易被忽略，值 3–5 人日）**：
1. **API Key 不能进客户端**：加一层自建代理（Cloudflare Worker / 轻量 FastAPI），做鉴权 + 限流 + 计费；否则 APK 反编译即泄露（现有 config.json 就是明文）。
2. **图片压缩与 token 控制**：上传前压到长边 1024–1280、JPEG q80，否则单图 token 翻倍、费用与延迟双升。
3. **超时/重试/降级**：VLM 超时 → 先返回 YOLO+规则库结论，异步补 VLM 结论。
4. **合规**：生成式 AI 输出需要内容标识与安全过滤（见 §6）。

---

## 5. 跨平台 UI 框架选型

### 5.1 明确结论：**Chaquopy 和 Kivy 都不能直接复用 PyQt5 代码**

**（1）Chaquopy（Android 内嵌 CPython）——不能复用 PyQt5**
- 官方 FAQ：Chaquopy 可配合 **任何**允许改 `build.gradle` 并能调用 Java 方法的框架（**含 Flutter / React Native**）；**不支持 iOS**（"Does Chaquopy support iOS? Not at the moment."）——chaquo.com/chaquopy/doc/current/faq.html【官方文档】
- Android 包源（chaquo.com/pypi-13.1/）目录中**不存在 pyqt5 / pyside** 条目；存在的是 numpy / opencv-python / pillow / pandas 等。**`torch` 的最后更新时间是 2021-05-22**，`torchvision` 2021-05-17【实测·官方索引】。
  → **推论：既不能跑 PyQt5，也不能用 pip 装新版 torch/ultralytics 做端侧推理。**
- minSdk ≥ 24；每个 ABI 增加数 MB；**APK split / app bundle 对 Chaquopy 原生组件帮助有限**，官方建议用 product flavor 砍 ABI【官方文档】。
- 可行用法：只把**不含 GUI 的纯 Python 业务逻辑**（诊断规则、prompt 拼装、结果格式化）搬过去，推理改成 TFLite/ONNX Runtime，UI 用 Kotlin/Flutter 写。

**（2）Kivy / python-for-android / Buildozer——不能复用 PyQt5**
- Kivy 的 Android 打包只有 Buildozer / python-for-android / Kivy Launcher 三条路，bootstrap 基于 **SDL2**（kivy.org/doc/stable/guide/packaging-android.html）【官方文档】；**没有 PyQt5 recipe**。
- 想用 PyQt5 上 Android 只剩 **pyqtdeploy + Qt for Android** 这条路：需要自己编 Qt、链路极重、生态稀薄；且 PyQt5 是 GPL/商业双许可，闭源商用需购买 Riverbank 商业授权。**【不确定】**（未找到 2026 年可用的 pyqtdeploy 成功案例；StackOverflow 相关提问已是多年前）
- 结论：**PyQt5 的窗口类（HomeWindow/ImageRecogWindow/CameraRecogWindow/ResultWindow/ApiSettingsWindow）必须全部重写。**

**（3）推荐选型（本项目）**

| 方案 | 是否复用 PyQt5 | 复用 Python 逻辑 | 工时倍率（相对单端原生=1.0） | 评价 |
| --- | --- | --- | --- | --- |
| **Kotlin 原生（Android）** | ❌ | 需重写（可对拍验证） | **1.0×** | NPU/相机/性能最省心；iOS 需另写 Swift |
| **Flutter** | ❌ | 需重写（Dart）；可通过 method channel 调 Kotlin 做推理 | **1.15–1.35×**（双端只需一套 UI） | **推荐**。Ultralytics 官方有 Flutter 插件（CoreML/LiteRT/QNN 都覆盖），端侧 YOLO 有现成路径 |
| **React Native** | ❌ | 同上（JS/TS） | 1.2–1.5× | 生态成熟，但 NPU 接入要自己写 native module |
| **Kivy/BeeWare** | ❌ | ✅ 理论可复用纯 Python 逻辑 | **2–3×** | UI 生态弱、包体大、相机/地图插件都要自己写；**不推荐** |
| **Chaquopy 混合（Kotlin UI + Python 逻辑）** | ❌ | ✅ 可复用非 GUI Python | **2.5–4×** | 打包/调试/ABI/包体成本高；**不推荐作为主线**，仅适合"Python 逻辑极重且必须原样保留"的场景 |
| **pyqtdeploy（PyQt5 → Android）** | ⚠️ 理论可以 | ✅ | **≥4×**【不确定】 | 工程风险极高，不建议 |

> 工时倍率的来源说明：Flutter/RN vs 原生"省一套代码库"的定性结论有公开讨论（如 tribulant.com 2026 年对比文），但**具体倍率是本报告基于经验给出的估计值，非权威统计**【不确定】。

---

## 6. 安卓上架与国内合规要点

### 6.1 时间线（这是排期里最不可压缩的部分）

| 事项 | 耗时 | 性质 | 来源 |
| --- | --- | --- | --- |
| **App 备案（工信部）** | 云厂商初审 **1–2 个工作日** → 省管局终审 **20 个工作日内** | 法定/官方口径 | support.huaweicloud.com/icp_faq/icp_05_0257.html【实测·官方】 |
| **软件著作权登记** | 受理后 **30–60 个工作日**（普通）；官方登记费自 2017 年起停征；加急另收费 | 代理商口径 | ayalm.com（2026-09）【二手】 |
| **算法备案** | 法规要求：提供服务之日起 **10 个工作日内**履行备案手续；审核时长实践通常 **1–3 个月** | 法规 + 律所解读 | weihenglaw.com《算法备案要点解析》；hankunlaw PDF【二手】 |
| **生成式 AI（大模型）备案** | 实践 **3 个多月**（某团队自述，8 个坑）；另有 **4–10 个月**区间说法 | 服务商/团队自述 | cloud.tencent.cn/developer/article/2598988【二手】 |
| **App Store 审核** | **1–3 个工作日**（厂商经验值）；Apple 官方"多数 24 小时内"的说法**未能抓取到官方页面原文**【不确定】 | 厂商经验 | ayalm.com【二手】 |
| **华为应用市场** | **1–5 个工作日**（重点看软著资质、内容合规） | 厂商经验 | ayalm.com【二手】 |
| **小米应用商店** | **1–5 个工作日** | 厂商经验 | ayalm.com【二手】 |
| **OPPO / vivo** | **1–7 个工作日** | 厂商经验 | ayalm.com【二手】 |
| **应用宝（腾讯）** | 公开资料多为流程说明，**未取到可靠时长数字**【不确定】；行业实践通常同量级（1–5 工作日） | — | — |

### 6.2 合规清单（按"必须做"排序）

1. **App 备案（强制）**：2023 年 8 月起，未备案 App 不得从事互联网信息服务。需先有 ICP 备案的域名/主体。
2. **隐私政策页（强制）**：国内商店要求**可公网访问的 URL**；列出权限用途（相机、存储、网络、定位）、第三方 SDK 清单（含 VLM API 服务商）、数据出境说明。被拒原因第一位就是"隐私政策缺失/不完整"。
3. **软件著作权证书**：多数国内安卓商店要求提供；**建议项目启动第一周就提交**（受理单可作为部分商店的先行材料）。
4. **AI 相关备案的判断（重要，且最容易踩）**：
   - 若**只调用已备案的第三方大模型 API**（百炼/DeepSeek），通常不需要自己做"大模型备案"，但需关注：**算法备案/深度合成备案**是否触发、以及**生成内容标识**义务。
   - 若**对模型做了微调/深度开发**（Fine-tune、改核心参数），按实践口径**需要备案**（来源：腾讯云文章"拿开源模型改了核心参数、做了微调的，都得备案"）。
   - **本项目风险点**：`best.pt` 是自训练模型 + 云端 VLM 生成诊断建议，属于"生成合成类"服务的边缘地带。**必须找合规顾问做一次正式判断**【不确定】。
5. **内容安全**：诊断结论涉及农药/植保建议，属敏感领域；建议在 UI 上明示"AI 建议，不替代农技人员"，并接入内容审核。
6. **相机/存储权限**：Android 13+ 用 `READ_MEDIA_IMAGES`（不要再申请 `READ_EXTERNAL_STORAGE`）；Chaquopy 官方也提示 API 29+ 分区存储不能直接读 sdcard，必须走系统文件选择器。
7. **多商店并行提交**：以最严格商店（通常华为）的标准准备材料，可少走 1–2 轮返工。

### 6.3 上架阶段的现实排期建议

- 首次上架预留 **2–3 周**（4–5 个商店并行 + 至少 1 轮被拒整改）。
- App 备案与软著**必须与开发并行启动**，否则会成为发布日期瓶颈。

---

## 7. 三档产品工期估算（1 名熟练移动开发者）

**估算假设（很重要）**：
- "熟练移动开发者" = 有 2–3 个已上线 App 经验，能独立完成 Android 端，熟悉相机/网络/本地存储，**但不懂模型转换与 NPU 调优**（这部分要现学，已计入）。
- 1 人日 = 8 小时有效工作；**不含** UI 视觉设计、后端从零建设、备案等待时间、以及被拒返工。
- 区间宽度按 **±30%** 理解；下文区间已包含常见返工。
- 默认 **只做 Android**；双端（含 iOS）请按 §5 建议 **×1.6–2.0**。

### 档位 A：轻量 MVP（拍照 → 上传云端推理 → 返回结果）

| 工作包 | 人日 |
| --- | --- |
| 项目脚手架、相机、权限（Android 13+） | 2–3 |
| 图片压缩、上传、云端 API 对接（含超时/重试/流式） | 3–5 |
| 结果页 UI + 历史记录（本地） | 3–4 |
| 设置页、隐私政策页、免责声明 | 1–2 |
| 打包、签名、上架材料准备与提交 | 2–4 |
| 真机测试与修复 | 2–3 |
| **合计** | **12–20 人日** |

- **日历**：3–5 周（含审核等待）
- **里程碑**：M1 相机+上传打通（D3）；M2 端到端出结果（D7）；M3 内测包（D12）；M4 提交上架（D16）；M5 上线（D20）
- **适用**：快速验证"农户是否愿意拍照问诊"；无端侧模型、无离线能力

### 档位 B：完整离线 App（端侧 YOLO + 端侧规则库 + 云端 VLM）

| 工作包 | 人日 |
| --- | --- |
| 模型转换/量化/NMS 与预处理实现、精度对拍验收 | 5–10 |
| 端侧推理集成（相机流、多线程、GPU/NPU 委托、机型降级） | 5–8 |
| 端侧规则库 + 诊断逻辑（Python → Kotlin/Dart 重写 + 对拍测试） | 5–10 |
| 云端 VLM 二次诊断与结果融合（含自建代理防 Key 泄露） | 4–6 |
| UI（首页/识别/相机/结果/知识/设置 6 个页面） | 8–12 |
| 本地存储、离线队列、弱网同步 | 3–5 |
| 性能与兼容性测试（多机型、发热、内存、冷启动） | 5–8 |
| 上架与合规（备案材料、隐私、权限） | 3–5 |
| **合计** | **35–60 人日** |

- **日历**：2–3 个月
- **里程碑**：M1 模型真机跑通且精度验收通过（1–2 周）；M2 端侧识别+规则库闭环（3–4 周）；M3 云端 VLM 融合（5–6 周）；M4 功能完备内测（7–8 周）；M5 提审上架（9–11 周）
- **关键风险**：模型转换与量化验收（M1）最容易延期；**务必先做 1 台目标机型的技术验证（3–5 人日）再承诺排期**

### 档位 C：完整 App + 地图/GIS + 用户体系 + 病害知识库

| 工作包 | 人日 |
| --- | --- |
| 档位 B 全部内容 | 35–60 |
| 账号体系（注册/登录/短信/第三方/Token 刷新/注销） | 5–8 |
| 后端（API、数据库、对象存储、运营后台）——若无现成后端 | 10–20 |
| 地图/GIS（地块绘制、坐标转换、打点、离线瓦片/轨迹） | 10–18 |
| 病害知识库（内容模型、检索、富文本/图文、离线缓存） | 8–15 |
| 数据看板/统计 | 4–6 |
| 埋点、崩溃监控、推送、灰度发布 | 4–6 |
| 多机型 QA + 性能专项 | 8–12 |
| 合规总包（App 备案、软著、隐私、AI 相关备案协调） | 5–10 |
| **合计** | **70–130 人日** |

- **日历（开发）**：3.5–5 个月；**上线日历**需再叠加备案等待 **1–3 个月**（App 备案 3–5 周 + 软著 6–12 周 + AI 相关备案 1–4 个月，可并行但要提前启动）
- **里程碑**：M1 骨架+账号（2 周）；M2 端侧识别闭环（5 周）；M3 GIS 地块（8 周）；M4 知识库+后台（11 周）；M5 灰度（13 周）；M6 全量上架（15–18 周）
- **前置条件**：若后端/知识库内容由他人提供，可减 20–35 人日；若从零建后端+内容，请按上限估

### 档位对比一览

| 档位 | 人日区间 | 日历（开发） | 日历（含合规等待） | 端侧 YOLO | 云端 VLM | GIS/用户体系 |
| --- | --- | --- | --- | --- | --- | --- |
| A 轻量 MVP | **12–20** | 3–4 周 | 3–5 周 | ❌ | ✅ | ❌ |
| B 完整离线 | **35–60** | 2–3 个月 | 3–4 个月 | ✅ | ✅ | ❌ |
| C 完整 App | **70–130** | 3.5–5 个月 | 5–8 个月 | ✅ | ✅ | ✅ |

---

## 8. 待补清单（Gap / 待补）

> 本轮为只读桌面调研，**本报告已落盘定稿**；下列为已知缺口，标注为「待补」，不影响已给结论的方向性。

| # | 待补项 | 现状 | 补齐方式 | 优先级 |
| --- | --- | --- | --- | --- |
| G1 | 本项目 `best.pt` 的 **端侧 INT8 实测 mAP 降幅** | 【待补】报告 §2.3 只有行业经验值 | Lead 侧已完成 ONNX 导出 + INT8 量化实测，需将其 mAP/耗时结果回填本节 | 高 |
| G2 | **中端安卓（骁龙 7 系 / 天玑 8000 系）CPU 与 GPU 路线**的 YOLOv8n 640 端到端耗时 | 【待补】仅有 NPU 官方数字，CPU/GPU 为推断 | 真机 3–5 人日 Spike | 高 |
| G3 | 端到端耗时口径统一（是否含 letterbox/NMS/绘制） | 【待补】各来源口径不同，已在 §2.2 提示 | 与 G2 同批实测时固定口径 | 高 |
| G4 | `qwen-vl-max` 现价与可用性；是否切 qwen3-vl 系列 | 【待补】只有 qwen-vl-plus 快照价 | 百炼控制台核对 | 中 |
| G5 | 本项目是否触发"生成式 AI 备案"的法律判断 | 【待补】非技术判断 | 合规顾问确认 | 中 |
| G6 | ExecuTorch 1.0 安卓端实际成熟度与 YOLO 实测 | 【待补】仅查得发布新闻标题 | 官方文档 + 真机 | 低 |
| G7 | 应用宝审核时长、Apple 官方审核时长原文 | 【待补】仅有厂商经验值 | 开发者后台实测记录 | 低 |

### 8.1 明确标注的不确定项（需要下一步验证）

1. **中端安卓（骁龙 7 系/天玑 8000 系）CPU/GPU 路线的 YOLOv8n 640 实测耗时**：只有 NPU 侧有官方数字，CPU/GPU 是外推。【须实测】
2. **YOLOv8/v11 INT8 量化后 mAP 下降的具体百分点**：官方只给定性描述；经验值 0.5–3 个百分点无一手来源。**必须用本项目验证集实测**。
3. **Chaquopy 在 2026 年是否已支持较新 torch / 是否有 PySide 轮子**：以官方索引为准，索引显示 torch 停在 2021；如需确认请查 chaquo 版本表。
4. **pyqtdeploy 部署 PyQt5 到 Android 的实际可行性（2026）**：仅找到多年前的问答，无成功案例；本报告倾向"不可行"。
5. **`qwen-vl-max` 当前官方价格与可用性**：未取到现价页面；百炼现主推 qwen3.x 系列。
6. **ExecuTorch 1.0 的实际成熟度与安卓端 YOLO 性能**：只查到发布新闻标题，未取到正文与实测。
7. **应用宝审核时长**、**Apple 官方"24 小时内"原文**：未取到官方页面。
8. **Paddle Lite 维护状态**：有"是否还在维护"的 issue 标题，未取到正文结论。
9. **本项目是否触发"生成式 AI 备案"**：属法律判断，需合规顾问确认。

## 9. 给项目组的落地建议（按优先级）

1. **先做 3–5 人日技术验证（Spike）**：在 1 台骁龙 7 系 + 1 台天玑 8000 系真机上，把 `best.pt` 导出 ONNX → LiteRT(w8a16) 与 NCNN(FP16/INT8) 两条路各跑通一次，记录**端到端耗时（含 NMS）与 mAP**。这个数据决定档位 B 能否成立。
2. **UI 框架选 Flutter**（或纯 Kotlin）：放弃一切"复用 PyQt5"的幻想，把 PyQt5 代码当作**需求说明书**而不是代码资产。
3. **VLM 走云端 + 自建代理**；端侧 VLM 作为高端机可选降级项，不进 MVP。
4. **合规并行启动**：第 1 周就提交软著与 App 备案，第 2 周确认 AI 备案口径。
5. **保留 Python 桌面端作为"训练/验证/对拍"工具**：用桌面端生成黄金输出，作为移动端重写逻辑的回归基准。

---

## 10. 参考来源（均为本次调研实际访问）

**端侧推理 / 模型导出**
- Ultralytics 官方文档：LiteRT 导出与实测 <https://docs.ultralytics.com/integrations/litert.md>
- Ultralytics 官方文档：Qualcomm QNN 导出与实测 <https://docs.ultralytics.com/integrations/qnn.md>
- Ultralytics 官方文档：CoreML/ANE 导出与实测 <https://docs.ultralytics.com/integrations/coreml.md>
- Ultralytics 官方文档：NCNN 导出 <https://docs.ultralytics.com/integrations/ncnn.md>
- Ultralytics 官方文档：MNN 导出（含 v8/v11 任务支持表） <https://docs.ultralytics.com/integrations/mnn.md>
- Ultralytics 官方文档：ExecuTorch 导出 <https://docs.ultralytics.com/integrations/executorch.md>
- Ultralytics 官方文档：ONNX 导出与 INT8 <https://docs.ultralytics.com/integrations/onnx.md>
- Ultralytics 官方文档：导出参数（nms / dynamic / quantize 语义） <https://docs.ultralytics.com/modes/export.md>
- Qualcomm AI Hub 模型卡：YOLOv8-Detection（各芯片 ms / 模型体积） <https://hf-mirror.com/qualcomm/YOLOv8-Detection>
- ONNX Runtime：Qualcomm QNN Execution Provider <https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider>
- 骁龙 865 上 YOLOv8 各尺寸 FPS（二手博客） <https://blog.csdn.net/ZhuBin365/article/details/157223786>
- 移动端推理引擎全景对比（ncnn/MNN/ORT/Paddle-Lite/MACE/TNN） <https://estelledc.github.io/rhino-bird-2026/research/mobile-inference/mobile-inference-landscape/>
- 端侧部署流水线与量化踩坑（ncnn2table / MNNConvert） <https://estelledc.github.io/rhino-bird-2026/research/mobile-inference/mobile-deployment-pipeline/>
- Paddle-Lite 转换 YOLOv8 的 SiLU 算子不支持问题 <https://blog.gitcode.com/da161a03187313f00a3f5675387a440c.html>

**端侧 / 云端 VLM**
- MiniCPM-V 2.6 模型卡（8.1B / 16.2 GB BF16） <https://hf-mirror.com/openbmb/MiniCPM-V-2_6>
- MiniCPM-V 4.0 端侧实测整理（iPhone 16 Pro Max <2s TTFT、>17 tok/s、OpenCompass 69.0） <https://blog.gitcode.com/fe0c1a7d90933bbbf9d86a4b2e912d0e.html>
- MiniCPM-V 4 硬件需求 <https://llmrun.dev/model/openbmb-minicpm-v-4>
- SmolVLM-Instruct 模型卡（2.246B / 4.49 GB BF16） <https://hf-mirror.com/HuggingFaceTB/SmolVLM-Instruct>
- Qwen2.5-VL-3B-Instruct 模型卡 <https://hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct>
- DeepSeek API 定价（deepseek-flash 支持 Vision） <https://api-docs.deepseek.com/quick_start/pricing/>
- 阿里云百炼 Qwen-VL 价格与限流 <https://help.aliyun.com/zh/model-studio/vision-white>
- 阿里云百炼模型总价目表 <https://help.aliyun.com/zh/model-studio/model-pricing>

**Python on Mobile / UI 框架**
- Chaquopy FAQ（不支持 iOS；可与 Flutter/RN 配合；建议用 TFLite 替代 TensorFlow） <https://chaquo.com/chaquopy/doc/current/faq.html>
- Chaquopy Gradle 插件（minSdk 24、ABI、包体） <https://chaquo.com/chaquopy/doc/current/android.html>
- Chaquopy Android wheel 索引（无 PyQt5；torch 停更于 2021-05-22） <https://chaquo.com/pypi-13.1/>
- Chaquopy 分区存储限制（API 29+） <https://chaquo.com/chaquopy/doc/current/faq.html>
- Kivy 官方：Android 打包（Buildozer / python-for-android / SDL2 bootstrap） <https://kivy.org/doc/stable/guide/packaging-android.html>
- Flutter / RN / 原生成本对比（2026，定性） <https://tribulant.com/blog/app-development/what-flutter-vs-react-native-vs-native-actually-costs-in-2026/>

**合规 / 上架**
- 华为云：App 备案审核时间（初审 1–2 工作日，管局 20 个工作日） <https://support.huaweicloud.com/icp_faq/icp_05_0257.html>
- 主流应用商店审核周期与材料（2026-09） <https://www.ayalm.com/app-shang-jia-shen-he-zhi-nan/>
- 软件著作权登记材料与周期（受理后 30–60 工作日） <https://www.ayalm.com/%e8%bd%af%e8%91%97%e7%94%b3%e8%af%b7%e9%9c%80%e8%a6%81%e4%bb%80%e4%b9%88%e6%9d%90%e6%96%99%ef%bc%9a%e8%bd%af%e4%bb%b6%e8%91%97%e4%bd%9c%e6%9d%83%e7%99%bb%e8%ae%b0%e6%b5%81%e7%a8%8b%e3%80%81%e5%ae%a1/>
- 大模型备案实践周期（3 个多月 / 4–10 个月） <https://cloud.tencent.cn/developer/article/2598988>
- 算法备案要点解析（律所） <https://www.weihenglaw.com/global-network/network-news/35>
- 《互联网信息服务算法推荐管理规定》全文 <https://www.gov.cn/zhengce/zhengceku/2022-01/04/content_5666429.htm>

> 声明：本报告为技术选型调研，非法律意见；工期为估算区间非承诺；所有【推断】与【不确定】项在用于立项决策前须做真机实测或官方确认。
