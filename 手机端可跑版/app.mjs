// app.mjs —— 手机端识别主逻辑（本地推理，照片不上传）
import * as ort from "./ort.wasm.min.mjs";
import { letterbox, postprocess, CLASS_NAMES } from "./detect-core.mjs";

const MODEL_URL = "./model.onnx";   // 由 best.pt 导出的 ONNX（与桌面端同一份权重）
// 加载器、wasm 与页面放在同一目录：ORT 会相对「加载器自身 URL」解析 wasm 文件名，
// 只要三者同级就不需要设 wasmPaths（设了反而容易多拼一层目录）。
ort.env.wasm.proxy = false;
// 未开启跨源隔离时必须单线程，否则 SharedArrayBuffer 不可用会直接报错
if (typeof self !== "undefined" && self.crossOriginIsolated && (navigator.hardwareConcurrency || 1) > 1) {
  ort.env.wasm.numThreads = 4;
}

const CLASS_COLORS = ["#e8590c", "#1971c2", "#0ca678", "#c92a2a", "#9c36b5", "#f08c00"];
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), statusCard = $("statusCard");

let session = null, treatment = null, imgBitmap = null, lastDets = [];

function setStatus(msg, isErr) {
  statusEl.innerHTML = msg;
  statusEl.className = "status" + (isErr ? " err" : "");
  statusCard.style.display = "";
}

async function init() {
  try {
    treatment = await (await fetch("./treatment.json")).json();
  } catch (e) {
    treatment = {};
  }
  try {
    setStatus("正在加载推理引擎…");
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    setStatus("模型已就绪（本地运行）。请选择照片或拍照。");
  } catch (e) {
    setStatus("模型加载失败：" + (e && e.message ? e.message : e) +
      "<br>请确认通过局域网地址访问（不要用 file:// 直接打开），并检查 vendor 与模型文件是否完整。", true);
  }
}

// 从 JPEG 绘制到 canvas，并强制缩放，避免处理手机原图（可能 4000px 宽）
const work = document.createElement("canvas");
const wctx = work.getContext("2d", { willReadFrequently: true });
const MAX_SIDE = 1280;

function drawToCanvas(bitmap) {
  const sw = bitmap.width, sh = bitmap.height;
  const scale = Math.min(1, MAX_SIDE / Math.max(sw, sh));
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  work.width = w; work.height = h;
  wctx.drawImage(bitmap, 0, 0, w, h);
  return { w, h };
}

async function runDetect(bitmap, conf) {
  const { w, h } = drawToCanvas(bitmap);
  const imageData = wctx.getImageData(0, 0, w, h);
  const rgb = new Uint8Array(w * h * 3);
  const src = imageData.data;
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    rgb[i * 3] = src[j]; rgb[i * 3 + 1] = src[j + 1]; rgb[i * 3 + 2] = src[j + 2];
  }
  const lb = letterbox(rgb, w, h, 640);
  const input = new ort.Tensor("float32", lb.data, [1, 3, 640, 640]);
  const t0 = performance.now();
  const out = await session.run({ [session.inputNames[0]]: input });
  const ms = performance.now() - t0;
  const o = out[session.outputNames[0]];
  const dets = postprocess(o.data, o.dims, lb, w, h, { conf });
  return { dets, ms, w, h };
}

function drawBoxes(bitmap, dets) {
  const cv = $("cv");
  cv.width = bitmap.width;
  cv.height = bitmap.height;
  const ctx = cv.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const sx = bitmap.width / work.width, sy = bitmap.height / work.height;
  const base = Math.max(2, Math.round(bitmap.width / 320));
  ctx.lineWidth = base * 2;
  ctx.font = "bold " + Math.round(bitmap.width / 22) + "px sans-serif";
  ctx.textBaseline = "top";
  for (const d of dets) {
    const x = d.box[0] * sx, y = d.box[1] * sy;
    const bw = (d.box[2] - d.box[0]) * sx, bh = (d.box[3] - d.box[1]) * sy;
    const color = CLASS_COLORS[d.cls % CLASS_COLORS.length];
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, bw, bh);
    const text = d.label + " " + (d.score * 100).toFixed(0) + "%";
    const tw = ctx.measureText(text).width;
    const th = Math.round(bitmap.width / 22) * 1.5;
    const ty = y - th > 0 ? y - th : y;
    ctx.fillStyle = color;
    ctx.fillRect(x, ty, tw + base * 4, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x + base * 2, ty + th * 0.16);
  }
  $("stage").style.display = "";
}

function renderResult(dets, ms, w, h) {
  if (!dets.length) {
    $("result").innerHTML = "<div class='kv'><b>未检出目标</b></div>" +
      "<div class='note'>当前阈值下没有检出。可尝试降低置信度阈值，或换一张目标更清晰、更近的照片。" +
      "注意：病害早期斑点多而小，远距离拍摄容易漏检。</div>";
    $("planCard").style.display = "none";
  } else {
    const rows = dets.map((d) =>
      "<div class='row'><span class='tag' style='color:" + CLASS_COLORS[d.cls % 6] + "'>" + d.label +
      "</span><span class='conf'>置信度 " + (d.score * 100).toFixed(1) + "%</span></div>").join("");
    $("result").innerHTML = "<div class='kv'><b>检出 " + dets.length + " 个目标</b>　" +
      "<span class='conf'>推理 " + ms.toFixed(0) + " ms · 处理尺寸 " + w + "×" + h + "</span></div>" + rows;
    renderPlan(dets[0]);
  }
  $("resultCard").style.display = "";
}

function renderPlan(top) {
  const info = treatment[top.label];
  if (!info) { $("planCard").style.display = "none"; return; }
  let html = "<div class='kv'><b>防治方案 · " + top.label + "</b>";
  html += "<div class='note' style='margin-top:4px'>" + (info.symptom || "") + "</div></div>";
  if (info.agri && info.agri.length) {
    html += "<div class='kv' style='margin-top:10px'><b>农业防治</b><ul>" +
      info.agri.map((s) => "<li>" + s + "</li>").join("") + "</ul></div>";
  }
  if (info.bio && info.bio.length) {
    html += "<div class='kv' style='margin-top:10px'><b>生物防治</b><ul>" +
      info.bio.map((s) => "<li>" + s + "</li>").join("") + "</ul></div>";
  }
  if (info.threshold) {
    html += "<div class='kv' style='margin-top:10px'><b>防治指标</b><div class='note' style='margin-top:2px'>" +
      info.threshold + "</div></div>";
  }
  if (info.drugs && info.drugs.length) {
    html += "<div class='kv' style='margin-top:12px'><b>推荐药剂</b></div>";
    for (const d of info.drugs) {
      html += "<div class='drug'><div class='nm'>" + d.name + "</div>" +
        "<div><span class='pill'>" + d.spec + "</span><span class='conf'>" + d.dose + "</span></div></div>";
    }
    html += "<div class='note'>用药请遵守当地农药标签与安全间隔期；同一药剂避免连续使用以防抗性。</div>";
  }
  $("plan").innerHTML = html;
  $("planCard").style.display = "";
}

// 解码照片：优先 createImageBitmap（快），失败则退回 <img> 元素。
// 用 <img> 路径时浏览器会自动应用 EXIF 方向，避免手机竖拍照片被转 90 度。
async function decodeImage(file) {
  // 先走 <img>：浏览器会按 EXIF 自动摆正手机竖拍照片，兼容性也最好
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    await new Promise((res, rej) => {
      img.onload = res;
      img.onerror = () => rej(new Error("图片解码失败：可能不是有效的图片文件"));
      img.src = url;
    });
    return await createImageBitmap(img);
  } catch (e) {
    return await createImageBitmap(file);   // 兜底
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function handleFile(file) {
  if (!file) return;
  const conf = parseInt($("conf").value, 10) / 100;
  try {
    setStatus("正在解码照片…");
    imgBitmap = await decodeImage(file);
    drawBoxes(imgBitmap, []);
    setStatus("正在识别…");
    const { dets, ms, w, h } = await runDetect(imgBitmap, conf);
    lastDets = dets;
    drawBoxes(imgBitmap, dets);
    renderResult(dets, ms, w, h);
    setStatus("识别完成。可继续选择下一张照片。");
  } catch (e) {
    setStatus("识别失败：" + (e && e.message ? e.message : e), true);
  }
}

$("pick").addEventListener("click", () => $("file").click());
$("shoot").addEventListener("click", () => $("camera").click());
$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
$("camera").addEventListener("change", (e) => handleFile(e.target.files[0]));
$("conf").addEventListener("input", (e) => {
  $("confv").textContent = (parseInt(e.target.value, 10) / 100).toFixed(2);
});
$("conf").addEventListener("change", () => reRun());

async function reRun() {
  if (!imgBitmap) return;
  const conf = parseInt($("conf").value, 10) / 100;
  setStatus("正在按新阈值重新识别…");
  const { dets, ms, w, h } = await runDetect(imgBitmap, conf);
  lastDets = dets;
  drawBoxes(imgBitmap, dets);
  renderResult(dets, ms, w, h);
  setStatus("识别完成。");
}

// 支持 ?auto=样本名 自动跑一张内置样图，便于无人值守自检（正常使用不需要）
const auto = new URLSearchParams(location.search).get("auto");
init().then(async () => {
  if (!auto) return;
  try {
    const blob = await (await fetch("./samples/" + auto + ".jpg")).blob();
    await handleFile(new File([blob], auto + ".jpg", { type: "image/jpeg" }));
  } catch (e) {
    setStatus("自动测试加载失败：" + (e && e.message ? e.message : e), true);
  }
});
