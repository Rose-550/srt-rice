# -*- coding: utf-8 -*-
"""电脑端同步工具（方案 C 的"边"）

职责：从小程序上传的中转服务拉取新记录 → 下载图片 → 本地 YOLO 推理 →
      追加到《点位表.csv》→ 之后跑你现成的 gen_points.py 就能出地图。

可以两种方式使用：
    1) 命令行：python sync_pc.py --once        （拉一次，适合定时任务/调试）
              python sync_pc.py --watch        （常驻，每 30 秒拉一次）
    2) 作为模块嵌入现有 PyQt5 应用（见文件末尾的嵌入示例注释）

依赖：标准库即可运行；推理部分会自动选择可用后端：
    - 优先 ultralytics（你 rice-gpu 环境里有，直接用 best.pt）
    - 其次 onnxruntime（用导出的 model.onnx）
    - 都没有则只下载存档、不推理（记录类别留空）
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------- 配置 ----------------
DEFAULT_CONFIG = {
    "relay_base": "http://127.0.0.1:9000",     # 中转服务地址（上云后改成 https://你的域名）
    "token": "",                                # 与服务器 RELAY_TOKEN 一致
    "interval": 30,                             # watch 模式轮询间隔（秒）
    "conf_threshold": 0.4,                      # 与桌面端/手机端保持一致
    "model_pt": "../best.pt",                   # ultralytics 权重
    "model_onnx": "../手机端可跑版/model.onnx",  # onnxruntime 模型
    "points_csv": "../AI+教育大赛/web地图demo/点位表.csv",
    "images_dir": "../result/云边识别",          # 与 gen_points.py 的约定一致（相对项目根目录）
    "project_root": "..",                       # 点位表里的图片路径按「项目根目录」解析
    "state_file": "data/sync_state.json",
}
CSV_FIELDS = ["选用", "图片相对路径", "类别", "置信度", "纬度", "经度", "时间", "备注"]


def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    p = path or os.path.join(ROOT, "config.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    for k in ("model_pt", "model_onnx", "points_csv", "images_dir", "state_file", "project_root"):
        if not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(ROOT, cfg[k]))
    return cfg


def load_state(cfg):
    p = cfg["state_file"]
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"cursor": 0.0, "seen_ids": []}


def save_state(cfg, state):
    os.makedirs(os.path.dirname(cfg["state_file"]), exist_ok=True)
    state["seen_ids"] = state["seen_ids"][-5000:]        # 防止无限增长
    with open(cfg["state_file"], "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)


# ---------------- 推理后端 ----------------
class Detector:
    """自动挑选可用推理后端。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.kind = None
        self.model = None
        self._names = None
        self._init_ultralytics() or self._init_onnx()

    def _init_ultralytics(self):
        pt = self.cfg["model_pt"]
        if not os.path.exists(pt):
            return False
        try:
            from ultralytics import YOLO
        except ImportError:
            return False
        try:
            self.model = YOLO(pt)
            self._names = self.model.names
            self.kind = "ultralytics"
            return True
        except Exception as e:
            print("[sync] ultralytics 加载失败：%s" % e)
            return False

    def _init_onnx(self):
        onnx_path = self.cfg["model_onnx"]
        if not os.path.exists(onnx_path):
            return False
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        try:
            self.model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.kind = "onnxruntime"
            return True
        except Exception as e:
            print("[sync] onnxruntime 加载失败：%s" % e)
            return False

    @property
    def available(self):
        return self.kind is not None

    def infer(self, image_path):
        """返回 [{'label','score','box'}...]；失败返回空列表。"""
        if not self.available:
            return []
        try:
            if self.kind == "ultralytics":
                res = self.model(image_path, conf=self.cfg["conf_threshold"], imgsz=640, verbose=False)[0]
                out = []
                for b in res.boxes:
                    out.append({
                        "label": self._names[int(b.cls.item())],
                        "score": round(float(b.conf.item()), 4),
                        "box": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                    })
                return out
            return self._infer_onnx(image_path)
        except Exception as e:
            print("[sync] 推理失败 %s：%s" % (os.path.basename(image_path), e))
            return []

    # ----- onnxruntime 路径：预处理必须与训练/手机端一致 -----
    def _infer_onnx(self, image_path):
        import numpy as np
        import cv2

        def imread_unicode(p):
            # cv2.imread 在含中文的路径下会失败
            return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

        img = imread_unicode(image_path)
        if img is None:
            return []
        h, w = img.shape[:2]
        dst = 640
        r = min(dst / w, dst / h)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        padx, pady = (dst - nw) // 2, (dst - nh) // 2
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((dst, dst, 3), dtype=np.uint8)
        canvas[pady:pady + nh, padx:padx + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]

        sess = self.model
        name = sess.get_inputs()[0].name
        pred = sess.run(None, {name: tensor})[0][0]        # (4+nc, anchors)
        nc = pred.shape[0] - 4
        anchors = pred.shape[1]
        conf = self.cfg["conf_threshold"]
        cands = []
        for i in range(anchors):
            scores = pred[4:, i]
            c = int(scores.argmax())
            s = float(scores[c])
            if s < conf:
                continue
            cx, cy, bw, bh = (float(pred[0, i]), float(pred[1, i]), float(pred[2, i]), float(pred[3, i]))
            cands.append((s, c, [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
        cands.sort(reverse=True, key=lambda t: t[0])

        def iou(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / ua if ua > 0 else 0

        keep = []
        for s, c, box in cands:
            if all(not (kc == c and iou(box, kb) > 0.45) for _, kc, kb in keep):
                keep.append((s, c, box))

        names = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉", "白叶枯病", "胡麻斑病", "稻瘟病"]
        out = []
        for s, c, box in keep:
            b = [(box[0] - padx) / r, (box[1] - pady) / r, (box[2] - padx) / r, (box[3] - pady) / r]
            b = [max(0.0, min(b[0], w)), max(0.0, min(b[1], h)), max(0.0, min(b[2], w)), max(0.0, min(b[3], h))]
            out.append({"label": names[c] if c < len(names) else str(c),
                        "score": round(s, 4), "box": [round(v, 1) for v in b]})
        return out


# ---------------- 中转交互 ----------------
def http_get(url, token="", timeout=60):
    req = urllib.request.Request(url)
    if token:
        req.add_header("X-Auth-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_get_json(url, token="", timeout=60):
    return json.loads(http_get(url, token, timeout).decode("utf-8"))


def fetch_new_records(cfg, state):
    q = urllib.parse.urlencode({"since": state["cursor"], "limit": 200})
    url = cfg["relay_base"].rstrip("/") + "/api/records?" + q
    data = http_get_json(url, cfg["token"])
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "接口返回异常"))
    return data.get("records", [])


def download_image(cfg, rec):
    rid = rec["id"]
    ext = rec.get("ext") or ".jpg"
    dst_dir = cfg["images_dir"]
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, rid + ext)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst                                   # 已下载过，幂等
    url = cfg["relay_base"].rstrip("/") + rec["image_url"]
    blob = http_get(url, cfg["token"], timeout=180)
    with open(dst, "wb") as fh:
        fh.write(blob)
    return dst


def ensure_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()


def append_csv(cfg, image_path, label, score, lat, lng, when, note):
    ensure_csv(cfg["points_csv"])
    # gen_points.py 约定：图片路径相对「项目根目录」而非 CSV 所在目录，写错会导致「图片不存在」
    rel = os.path.relpath(image_path, cfg["project_root"]).replace("\\", "/")
    row = {
        "选用": "是",
        "图片相对路径": rel,
        "类别": label,
        "置信度": ("%.3f" % score) if score is not None else "",
        "纬度": lat if lat is not None else "",
        "经度": lng if lng is not None else "",
        "时间": when,
        "备注": note,
    }
    with open(cfg["points_csv"], "a", encoding="utf-8-sig", newline="") as fh:
        csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)
    return row


def process_once(cfg, state, detector, verbose=True):
    """拉取一轮。返回本轮处理的记录数。"""
    records = fetch_new_records(cfg, state)
    seen = set(state["seen_ids"])
    todo = []
    for rec in records:
        if rec["id"] in seen:
            continue
        if not rec.get("uploaded"):
            continue                                  # 图片还没传完，下轮再来
        todo.append(rec)

    if verbose:
        print("[sync] 拉到 %d 条，其中待处理 %d 条" % (len(records), len(todo)))

    done = 0
    for rec in todo:
        rid = rec["id"]
        try:
            img = download_image(cfg, rec)
        except Exception as e:
            print("[sync] 下载失败 %s：%s" % (rid, e))
            continue
        dets = []
        if detector.available:
            dets = detector.infer(img)
        if dets:
            top = max(dets, key=lambda d: d["score"])
            label, score = top["label"], top["score"]
            extra = "检出%d个" % len(dets)
            if len(dets) > 1:
                extra += "；最高：" + top["label"]
        else:
            label, score = "", None
            extra = "未检出" if detector.available else "未推理（无可用后端）"

        note = "小程序上传 %s" % extra
        if rec.get("address"):
            note += " @" + rec["address"]
        row = append_csv(cfg, img, label, score, rec.get("lat"), rec.get("lng"),
                         rec.get("created_at_str") or time.strftime("%Y-%m-%d %H:%M:%S"), note)
        seen.add(rid)
        done += 1
        if verbose:
            print("[sync]   %s  %s  %s  %.0f%%  %s" % (
                rid, os.path.basename(img), label or "(无)", (score or 0) * 100, extra))

    # 游标推进到本轮最大时间戳；用 >= 减一个极小量避免同秒记录被跳过
    if records:
        state["cursor"] = max(r.get("created_at", 0) for r in records)
    state["seen_ids"] = list(seen)
    save_state(cfg, state)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--once", action="store_true", help="只拉取一轮")
    ap.add_argument("--watch", action="store_true", help="常驻轮询")
    ap.add_argument("--loop", action="store_true", help="连续拉取直到没有新记录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    state = load_state(cfg)
    print("中转服务  : %s" % cfg["relay_base"])
    print("点位表    : %s" % cfg["points_csv"])
    print("图片存档  : %s" % cfg["images_dir"])
    detector = Detector(cfg)
    print("推理后端  : %s" % (detector.kind if detector.available else "无（仅存档）"))
    print("")

    if args.watch:
        print("[sync] 常驻模式，每 %d 秒拉取一次（Ctrl+C 停止）" % cfg["interval"])
        while True:
            try:
                process_once(cfg, state, detector)
            except Exception as e:
                print("[sync] 本轮失败：%s" % e)
            time.sleep(cfg["interval"])
    elif args.loop:
        total = 0
        while True:
            n = process_once(cfg, state, detector)
            total += n
            if n == 0:
                break
        print("\n[sync] 共处理 %d 条" % total)
    else:
        process_once(cfg, state, detector)


if __name__ == "__main__":
    main()

# ---------------- 嵌入 PyQt5 的示例（放进你现有应用） ----------------
# import threading, time
# from PyQt5.QtCore import QObject, pyqtSignal, QTimer
#
# class CloudSyncWorker(QObject):
#     """在后台线程拉取；不要在 Qt 主线程里发网络请求。"""
#     new_record = pyqtSignal(dict)      # 每处理一条就发一次，主窗口收到后刷新列表/地图
#     failed = pyqtSignal(str)
#
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.state = load_state(cfg)
#         self.detector = Detector(cfg)
#         self._stop = False
#
#     def run(self):
#         while not self._stop:
#             try:
#                 records = fetch_new_records(self.cfg, self.state)
#                 for rec in records:
#                     if rec["id"] in set(self.state["seen_ids"]) or not rec.get("uploaded"):
#                         continue
#                     img = download_image(self.cfg, rec)
#                     dets = self.detector.infer(img) if self.detector.available else []
#                     top = max(dets, key=lambda d: d["score"]) if dets else None
#                     append_csv(self.cfg, img, top["label"] if top else "",
#                                top["score"] if top else None,
#                                rec.get("lat"), rec.get("lng"),
#                                rec.get("created_at_str"), "小程序上传")
#                     self.state["seen_ids"].append(rec["id"])
#                     self.new_record.emit(rec)
#                 if records:
#                     self.state["cursor"] = max(r["created_at"] for r in records)
#                 save_state(self.cfg, self.state)
#             except Exception as e:
#                 self.failed.emit(str(e))
#             time.sleep(self.cfg["interval"])
#
# 在主窗口里这样接：
#     self._sync_thread = QThread()
#     self._worker = CloudSyncWorker(load_config())
#     self._worker.moveToThread(self._sync_thread)
#     self._sync_thread.started.connect(self._worker.run)
#     self._worker.new_record.connect(self.on_cloud_record)   # 你的刷新槽函数
#     self._sync_thread.start()
