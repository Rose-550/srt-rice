"""Confidence-threshold scan for the v8n val predictions.

Ground truth  : datasets/dataset_fieldlike/labels/val/*.txt   -> 5 cols  (cls cx cy w h)
Predictions   : runs/predict/pred_val_v8n/labels/*.txt -> 6 cols (cls cx cy w h conf)

Method: match every prediction against ground truth ONCE, greedily by descending
confidence, class-aware, IoU >= 0.5 (pixel coords - the images are not square, so
matching in normalized coords would distort IoU). Then a confidence threshold only
changes which already-matched predictions are counted, so the whole scan is free.

Outputs: checks/pred_lowconf_audit/threshold_scan_results.csv
         checks/pred_lowconf_audit/threshold_scan_best.csv
"""
import os
import csv
import glob
from collections import defaultdict

from PIL import Image

ROOT = os.path.join(os.environ["USERPROFILE"], "Desktop", "rice")
GT_LBL = os.path.join(ROOT, "datasets", "dataset_fieldlike", "labels", "val")
IMG = os.path.join(ROOT, "datasets", "dataset_fieldlike", "images", "val")
PRED_LBL = os.path.join(ROOT, "runs", "predict", "pred_val_v8n", "labels")
OUT = os.path.join(ROOT, "checks", "pred_lowconf_audit")

NAMES = {0: "稻纵卷叶螟", 1: "稻水象甲", 2: "稻叶蝉",
         3: "白叶枯病", 4: "胡麻斑病", 5: "稻瘟病"}
IOU_MIN = 0.5
THRESHOLDS = [round(0.01 * i, 2) for i in range(1, 100)]  # 0.01 .. 0.99


def read_gt(path):
    """-> [(cls, cx, cy, w, h)]  (5-col YOLO ground truth)"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            out.append((int(float(p[0])), float(p[1]), float(p[2]),
                        float(p[3]), float(p[4])))
    return out


def read_pred(path):
    """-> [(cls, cx, cy, w, h, conf)]  (6-col, conf kept)"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 6:
                continue
            out.append((int(float(p[0])), float(p[1]), float(p[2]),
                        float(p[3]), float(p[4]), float(p[5])))
    return out


def to_px(b, W, H):
    _, cx, cy, w, h = b[:5]
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---- collect matched predictions -------------------------------------
gt_count = defaultdict(int)                 # class -> n ground truth
matched = []                                # (cls, conf, is_tp)
stems = sorted({os.path.splitext(os.path.basename(f))[0]
                for f in glob.glob(os.path.join(PRED_LBL, "*.txt"))})

for stem in stems:
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        c = os.path.join(IMG, stem + ext)
        if os.path.exists(c):
            img_path = c
            break
    if img_path is None:
        continue
    W, H = Image.open(img_path).size

    gt = read_gt(os.path.join(GT_LBL, stem + ".txt"))
    pred = read_pred(os.path.join(PRED_LBL, stem + ".txt"))
    for b in gt:
        gt_count[b[0]] += 1

    gt_px = [(b[0], to_px(b, W, H)) for b in gt]
    used = [False] * len(gt_px)

    for b in sorted(pred, key=lambda x: -x[5]):
        box = to_px(b, W, H)
        best, best_j = 0.0, -1
        for j, (gcls, gbox) in enumerate(gt_px):
            if used[j] or gcls != b[0]:
                continue
            v = iou(box, gbox)
            if v > best:
                best, best_j = v, j
        if best >= IOU_MIN:
            used[best_j] = True
            matched.append((b[0], b[5], True))
        else:
            matched.append((b[0], b[5], False))

total_gt = sum(gt_count.values())

# ---- scan -------------------------------------------------------------
rows = []
for t in THRESHOLDS:
    tp = fp = 0
    per = {c: [0, 0] for c in range(6)}
    for cls, conf, is_tp in matched:
        if conf < t:
            continue
        if is_tp:
            tp += 1
            per[cls][0] += 1
        else:
            fp += 1
            per[cls][1] += 1
    fn = total_gt - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / total_gt if total_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rows.append(("all", t, tp, fp, fn, prec, rec, f1))
    mps, mrs, mfs = [], [], []
    for c in range(6):
        ctp, cfp = per[c]
        cfn = gt_count[c] - ctp
        cp = ctp / (ctp + cfp) if ctp + cfp else 0.0
        cr = ctp / gt_count[c] if gt_count[c] else 0.0
        cf = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
        rows.append((c, t, ctp, cfp, cfn, cp, cr, cf))
        mps.append(cp)
        mrs.append(cr)
        mfs.append(cf)
    # macro = unweighted mean over the 6 classes; this is the convention
    # Ultralytics reports precision/recall in, and it is NOT the pooled figure.
    rows.append(("macro", t, tp, fp, fn,
                 sum(mps) / 6, sum(mrs) / 6, sum(mfs) / 6))

with open(os.path.join(OUT, "threshold_scan_results.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["class", "class_name", "conf_threshold", "TP", "FP", "FN",
                "precision", "recall", "F1"])
    for cls, t, tp, fp, fn, p, r, f in rows:
        nm = cls if isinstance(cls, str) else NAMES[cls]
        w.writerow([cls, nm, "%.2f" % t, tp, fp, fn,
                    "%.4f" % p, "%.4f" % r, "%.4f" % f])

# ---- best F1 per class ------------------------------------------------
best = []
GROUPS = ["all", "macro"] + list(range(6))
for cls in GROUPS:
    sub = [r for r in rows if r[0] == cls]
    b = max(sub, key=lambda r: r[7])
    best.append(b)
    # also the F1 at the conventional 0.25 threshold
    at25 = min(sub, key=lambda r: abs(r[1] - 0.25))

with open(os.path.join(OUT, "threshold_scan_best.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["class", "class_name", "best_conf", "TP", "FP", "FN",
                "precision", "recall", "F1",
                "F1_at_0.25", "P_at_0.25", "R_at_0.25"])
    for (cls, t, tp, fp, fn, p, r, f), at25 in zip(best, [min([x for x in rows if x[0] == c],
                                                              key=lambda x: abs(x[1] - 0.25))
                                                          for c in GROUPS]):
        nm = cls if isinstance(cls, str) else NAMES[cls]
        w.writerow([cls, nm, "%.2f" % t, tp, fp, fn, "%.4f" % p, "%.4f" % r, "%.4f" % f,
                    "%.4f" % at25[7], "%.4f" % at25[5], "%.4f" % at25[6]])

# ---- report -----------------------------------------------------------
print("images with predictions :", len(stems))
print("ground-truth boxes      :", total_gt, dict((NAMES[c], gt_count[c]) for c in range(6)))
print("predictions             :", len(matched))
print()
print("=== best-F1 threshold per class ===")
print("%-12s %6s %5s %5s %5s %8s %8s %8s" % ("class", "conf", "TP", "FP", "FN", "P", "R", "F1"))
for (cls, t, tp, fp, fn, p, r, f), at25 in zip(best, [min([x for x in rows if x[0] == c],
                                                          key=lambda x: abs(x[1] - 0.25))
                                                      for c in GROUPS]):
    nm = cls if isinstance(cls, str) else NAMES[cls]
    print("%-12s %6.2f %5d %5d %5d %8.4f %8.4f %8.4f" % (nm, t, tp, fp, fn, p, r, f))
print()
print("=== overall (all classes) vs threshold ===")
print("%6s %6s %6s %6s %8s %8s %8s" % ("conf", "TP", "FP", "FN", "P", "R", "F1"))
for r in rows:
    if r[0] == "all" and any(abs(r[1] - k) < 1e-9 for k in
                             (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)):
        print("%6.2f %6d %6d %6d %8.4f %8.4f %8.4f" % (r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
print()
print("out:", OUT)
