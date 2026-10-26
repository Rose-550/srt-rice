// detect-core.mjs —— 推理后处理核心（浏览器与 Node 共用同一份代码）
// 输入约定：像素为 RGB 通道顺序，归一化到 [0,1]，尺寸 640x640（YOLOv8n 训练时的输入约定）

export const CLASS_NAMES = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉", "白叶枯病", "胡麻斑病", "稻瘟病"];
export const CLASS_NAMES_EN = ["leaf_roller", "water_weevil", "leafhopper", "bacterial_blight", "brown_spot", "rice_blast"];
export const DEFAULT_CONF = 0.4;   // 与 rice_dp1_optimized.py 的 CONF_THRESHOLD 保持一致
export const DEFAULT_IOU = 0.45;

/**
 * 等比缩放并居中填充到 dst x dst，返回 NCHW float32 张量数据。
 * 采用双线性插值，以便与 OpenCV 参考实现逐像素对比。
 * @param {Uint8ClampedArray|Uint8Array} rgb 长度 w*h*3 的 RGB 像素
 */
export function letterbox(rgb, w, h, dst = 640) {
  const r = Math.min(dst / w, dst / h);
  const nw = Math.max(1, Math.round(w * r));
  const nh = Math.max(1, Math.round(h * r));
  const padX = Math.floor((dst - nw) / 2);
  const padY = Math.floor((dst - nh) / 2);
  const area = dst * dst;
  const out = new Float32Array(3 * area);
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(h - 1, Math.max(0, (y + 0.5) / r - 0.5));
    const y0 = Math.floor(sy), y1 = Math.min(h - 1, y0 + 1), fy = sy - y0;
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(w - 1, Math.max(0, (x + 0.5) / r - 0.5));
      const x0 = Math.floor(sx), x1 = Math.min(w - 1, x0 + 1), fx = sx - x0;
      const o = (y + padY) * dst + (x + padX);
      for (let c = 0; c < 3; c++) {
        const p00 = rgb[(y0 * w + x0) * 3 + c];
        const p01 = rgb[(y0 * w + x1) * 3 + c];
        const p10 = rgb[(y1 * w + x0) * 3 + c];
        const p11 = rgb[(y1 * w + x1) * 3 + c];
        const top = p00 + (p01 - p00) * fx;
        const bot = p10 + (p11 - p10) * fx;
        out[c * area + o] = (top + (bot - top) * fy) / 255;
      }
    }
  }
  return { data: out, r, padX, padY, nw, nh };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union <= 0 ? 0 : inter / union;
}

/**
 * 解析 YOLOv8 输出 (1, 4+nc, anchors)：前 4 行是 cx,cy,w,h，其后每行是一类的分数。
 * 模型自身不含 NMS，必须在这里做。
 */
export function parseOutput(data, dims, opts = {}) {
  const conf = opts.conf ?? DEFAULT_CONF;
  const iouThr = opts.iou ?? DEFAULT_IOU;
  const nc = opts.numClasses ?? CLASS_NAMES.length;
  const anchors = dims[dims.length - 1];
  const cand = [];
  for (let i = 0; i < anchors; i++) {
    let best = 0, bestC = -1;
    for (let c = 0; c < nc; c++) {
      const s = data[(4 + c) * anchors + i];
      if (s > best) { best = s; bestC = c; }
    }
    if (bestC < 0 || best < conf) continue;
    const cx = data[i], cy = data[anchors + i];
    const bw = data[2 * anchors + i], bh = data[3 * anchors + i];
    cand.push({ box: [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], score: best, cls: bestC });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of cand) {
    let ok = true;
    for (const k of keep) {
      if (k.cls === b.cls && iou(k.box, b.box) > iouThr) { ok = false; break; }
    }
    if (ok) keep.push(b);
  }
  return keep;
}

/** 把 640 坐标系里的框映射回原图坐标，并做边界裁剪 */
export function toOriginal(keep, lb, w, h) {
  return keep.map((d) => {
    const b = [
      (d.box[0] - lb.padX) / lb.r,
      (d.box[1] - lb.padY) / lb.r,
      (d.box[2] - lb.padX) / lb.r,
      (d.box[3] - lb.padY) / lb.r,
    ].map((v) => Math.max(0, Math.min(v, v < 0 ? 0 : 1e9)));
    return {
      cls: d.cls,
      label: CLASS_NAMES[d.cls],
      labelEn: CLASS_NAMES_EN[d.cls],
      score: d.score,
      box: [b[0], b[1], b[2], b[3]].map((v, i) => (i % 2 === 0 ? Math.min(v, w) : Math.min(v, h))),
    };
  });
}

/** 完整后处理：张量 -> 原图坐标下的检出框 */
export function postprocess(data, dims, lb, w, h, opts = {}) {
  return toOriginal(parseOutput(data, dims, opts), lb, w, h);
}
