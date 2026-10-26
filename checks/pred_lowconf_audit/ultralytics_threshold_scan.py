"""Confidence-threshold scan using Ultralytics' OWN matching.

Instead of re-implementing IoU matching, this runs Ultralytics' DetectionValidator
once (conf=0.001, so the full prediction tail is captured), then reads the
validator's own per-prediction TP/FP bookkeeping from `validator.stats`.

That makes the TP/FP determination byte-for-byte the same as what produces the
numbers in results.csv, so the only remaining difference is aggregation
convention (micro vs macro), which is reported explicitly.

Outputs: checks/pred_lowconf_audit/ultralytics_threshold_scan.csv
         checks/pred_lowconf_audit/ultralytics_threshold_best.csv
"""
import os
import csv

import numpy as np
from ultralytics import YOLO

ROOT = os.path.join(os.environ["USERPROFILE"], "Desktop", "rice")
MODEL = os.path.join(ROOT, "runs", "detect", "yolov8n_clean", "weights", "best.pt")
DATA = os.path.join(ROOT, "dataset_fieldlike", "dataset.yaml")
OUT = os.path.join(ROOT, "checks", "pred_lowconf_audit")

NAMES = {0: "稻纵卷叶螟", 1: "稻水象甲", 2: "稻叶蝉",
         3: "白叶枯病", 4: "胡麻斑病", 5: "稻瘟病"}
IOU_COL = 0  # column 0 of the (N, 10) tp array == IoU 0.5

print("running Ultralytics val (conf=0.001, workers=0) ...", flush=True)
model = YOLO(MODEL)

# Two gotchas, both verified empirically:
#  1. Model.val() forces rect=True. Without it the numbers do NOT reproduce the
#     ones in results.csv (mp 0.883 vs 0.923 etc), so set it explicitly.
#  2. In 8.4 the raw per-prediction arrays live in validator.metrics.stats, and
#     DetectionValidator.get_stats() calls metrics.clear_stats() before returning.
#     So snapshot them BEFORE delegating to super().
from ultralytics.models.yolo.detect import DetectionValidator


class CapturingValidator(DetectionValidator):
    def get_stats(self):
        st = getattr(self.metrics, "stats", None)
        if isinstance(st, dict):
            cap = {}
            for k, v in st.items():
                try:
                    cap[k] = np.concatenate(v) if len(v) else np.zeros((0,))
                except Exception:
                    cap[k] = v
            self.captured = cap
        return super().get_stats()


vargs = {**model.overrides, "data": DATA, "split": "val", "imgsz": 640,
         "batch": 16, "device": 0, "conf": 0.001, "iou": 0.7,
         "rect": True, "mode": "val",
         "plots": False, "verbose": False, "workers": 0,
         "project": OUT, "name": "val_official", "exist_ok": True}
validator = CapturingValidator(args=vargs, _callbacks=model.callbacks)
validator(model=model.model)
print("val done.", flush=True)

res = validator.metrics

box = res.box
print()
print("=== Ultralytics 官方口径 (results.csv 里的就是这些) ===")
print("  mp (macro precision) = %.4f" % box.mp)
print("  mr (macro recall)    = %.4f" % box.mr)
print("  map50                = %.4f" % box.map50)
print("  map50-95             = %.4f" % box.map)
print("  per-class P =", np.round(box.p, 4).tolist())
print("  per-class R =", np.round(box.r, 4).tolist())

# ---- pull the validator's own TP/FP bookkeeping ------------------------
stats = validator.captured


def grab(key):
    v = stats[key]
    if isinstance(v, list):
        v = np.concatenate(v, 0) if v else np.zeros((0,), dtype=object)
    return np.asarray(v)


tp_all = grab("tp")
conf = grab("conf").astype(float)
pcls = grab("pred_cls").astype(int)
tcls = grab("target_cls").astype(int)

print()
print("  validator.stats shapes: tp=%s conf=%s pred_cls=%s target_cls=%s"
      % (tp_all.shape, conf.shape, pcls.shape, tcls.shape))

tp05 = tp_all[:, IOU_COL].astype(bool)
is_fp = ~tp05

uniq, counts = np.unique(tcls, return_counts=True)
gt_count = dict(zip(uniq.tolist(), counts.tolist()))
classes = sorted(gt_count)
total_gt = int(len(tcls))

print("  predictions = %d   ground truth = %d" % (len(conf), total_gt))

# ---- sanity: reproduce Ultralytics' micro numbers at its own cut ------
# Ultralytics picks the operating point that maximises the MEAN F1 over classes
# on an interpolated PR curve. We can't read that curve directly, so instead we
# report macro/micro over an explicit confidence grid and show where the peak is.

THRESHOLDS = [round(0.01 * i, 2) for i in range(1, 100)]
rows = []

for t in THRESHOLDS:
    keep = conf >= t
    tp_c = {}
    fp_c = {}
    for c in classes:
        sel = keep & (pcls == c)
        tp_c[c] = int(np.sum(sel & tp05))
        fp_c[c] = int(np.sum(sel & is_fp))

    tp_micro = sum(tp_c.values())
    fp_micro = sum(fp_c.values())
    fn_micro = total_gt - tp_micro
    p_micro = tp_micro / (tp_micro + fp_micro) if tp_micro + fp_micro else 0.0
    r_micro = tp_micro / total_gt if total_gt else 0.0
    f_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if p_micro + r_micro else 0.0

    ps, rs, fs = [], [], []
    for c in classes:
        tpc, fpc = tp_c[c], fp_c[c]
        fnc = gt_count[c] - tpc
        pc = tpc / (tpc + fpc) if tpc + fpc else 0.0
        rc = tpc / gt_count[c] if gt_count[c] else 0.0
        fc = 2 * pc * rc / (pc + rc) if pc + rc else 0.0
        rows.append((c, t, tpc, fpc, fnc, pc, rc, fc))
        ps.append(pc)
        rs.append(rc)
        fs.append(fc)

    rows.append(("micro", t, tp_micro, fp_micro, fn_micro, p_micro, r_micro, f_micro))
    rows.append(("macro", t, tp_micro, fp_micro, fn_micro,
                 float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))))

with open(os.path.join(OUT, "ultralytics_threshold_scan.csv"), "w",
          newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["class", "class_name", "conf_threshold", "TP", "FP", "FN",
                "precision", "recall", "F1"])
    for cls, t, tp, fp, fn, p, r, f in rows:
        nm = cls if isinstance(cls, str) else NAMES[cls]
        w.writerow([cls, nm, "%.2f" % t, tp, fp, fn,
                    "%.4f" % p, "%.4f" % r, "%.4f" % f])

GROUPS = ["micro", "macro"] + classes
best = {}
for cls in GROUPS:
    sub = [x for x in rows if x[0] == cls]
    best[cls] = max(sub, key=lambda x: x[7])

with open(os.path.join(OUT, "ultralytics_threshold_best.csv"), "w",
          newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["class", "class_name", "best_conf", "TP", "FP", "FN",
                "precision", "recall", "F1"])
    for cls in GROUPS:
        _, t, tp, fp, fn, p, r, f = best[cls]
        nm = cls if isinstance(cls, str) else NAMES[cls]
        w.writerow([cls, nm, "%.2f" % t, tp, fp, fn,
                    "%.4f" % p, "%.4f" % r, "%.4f" % f])

print()
print("=== best-F1 threshold per group (Ultralytics matching) ===")
print("%-10s %6s %6s %6s %6s %8s %8s %8s"
      % ("group", "conf", "TP", "FP", "FN", "P", "R", "F1"))
for cls in GROUPS:
    _, t, tp, fp, fn, p, r, f = best[cls]
    nm = cls if isinstance(cls, str) else NAMES[cls]
    print("%-10s %6.2f %6d %6d %6d %8.4f %8.4f %8.4f" % (nm, t, tp, fp, fn, p, r, f))

print()
print("=== micro vs threshold ===")
print("%6s %6s %6s %6s %8s %8s %8s" % ("conf", "TP", "FP", "FN", "P", "R", "F1"))
for x in rows:
    if x[0] == "micro" and any(abs(x[1] - k) < 1e-9 for k in
                               (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)):
        print("%6.2f %6d %6d %6d %8.4f %8.4f %8.4f" % (x[1], x[2], x[3], x[4], x[5], x[6], x[7]))

print()
print("=== macro vs threshold ===")
print("%6s %8s %8s %8s" % ("conf", "P", "R", "F1"))
for x in rows:
    if x[0] == "macro" and any(abs(x[1] - k) < 1e-9 for k in
                               (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)):
        print("%6.2f %8.4f %8.4f %8.4f" % (x[1], x[5], x[6], x[7]))

print()
print("out:", OUT)
