# 稻视界 · 手机端可跑版

把训练好的 YOLOv8n 权重（`best.pt`）放到手机浏览器里**本地推理**，不装 App、不传照片到服务器。

## 三步跑起来

1. 电脑和手机连**同一个 WiFi**。
2. 在电脑上启动服务器（用你跑训练的 rice-gpu 环境即可，或用任意 Python 3）：

   ```
   cd 手机端可跑版
   python serve.py
   ```

   终端会打印两个地址，例如：

   ```
   本机自测 : http://127.0.0.1:8765/
   手机访问 : http://192.168.1.23:8765/
   ```

3. 手机浏览器打开**手机访问**那个地址，点「拍照识别」即可。

> 首次打开需要下载推理引擎（约 11 MB）和模型（约 12 MB），建议在 WiFi 下进行；之后浏览器会缓存。
> 注意：不要用 `file://` 直接双击 index.html，那样浏览器会限制模块与 WASM 加载，必须通过 `serve.py` 访问。

## 已验证的事实

这一版不是「大概能跑」，而是实测过的（headless Chromium + 真实模型 + 真实样本图）：

| 验证项 | 结果 |
|---|---|
| 浏览器内加载 WASM 推理引擎 | 成功，跨源隔离生效（多线程可用） |
| 浏览器内加载 ONNX 模型 | 成功，输入 `images` / 输出 `output0` |
| 真实图片推理 | 成功，3/4 样本给出正确类别 |
| 单帧推理耗时（桌面 CPU，WASM） | 65–98 ms |
| 预处理与 OpenCV 一致性 | 几何完全一致，张量最大偏差 0.003 |
| 检测框叠加渲染 | 已人工核对，框落在真实病斑上 |

自检输出留档见 `selftest-result.txt`，渲染结果留档见 `annotated-overlay.png`。

想重新自检：浏览器打开 `http://<你的IP>:8765/selftest.html`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `serve.py` | 局域网服务器（含跨源隔离响应头，WASM 多线程需要） |
| `index.html` | 界面 |
| `app.mjs` | 主逻辑：选图/拍照 → 预处理 → 推理 → 画框 → 出防治方案 |
| `detect-core.mjs` | **核心算法**：letterbox、NMS、坐标反映射（与 Node 验证脚本共用同一份） |
| `model.onnx` | 由 `best.pt` 导出的模型（opset 19，静态 640×640） |
| `treatment.json` | 从 `rice_dp1_optimized.py` 的 `TREATMENT_DB` 自动抽取，6 类防治方案 |
| `ort.wasm.min.mjs` | onnxruntime-web 运行时（onnxruntime-web 1.23.0） |
| `ort-wasm-simd-threaded.*` | WASM 推理内核 |
| `samples/` | 用于自检的 4 张样本图 |
| `selftest.html` | 无人值守自检页，自动跑 3 张样本并打印结果 |
| `vendor/vendor/ort-wasm-simd-threaded.mjs` | 兼容垫片（老缓存路径用，可留可删） |

## 已知限制

- 这一版是**浏览器 Demo**，不是 APK。要变成能装的应用（端侧实时识别、离线可用）需要另做原生/Flutter 工程——那部分工时见上一轮评估。
- 单张推理 65–98 ms 是**桌面 CPU** 的数字；手机端请以真机实测为准（会慢一些，但可接受）。
- 4 张样本里有 1 张（`brown_spot_0`）没检出，符合你实验记录里「病害类召回偏弱」的结论。**手机实拍照片需要重新标定置信度阈值**。
- 界面里的置信度阈值滑杆改完会自动重跑，方便在手机上现场调阈值。
