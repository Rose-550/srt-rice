"""Per-image TP/FP/FN diagnosis for the blight class (class 3).

Greedy IoU>=0.5 matching against ground truth, in pixel coordinates
(images are not square, so normalized-coord IoU would be distorted).

Output: checks/pred_lowconf_audit/blight_image_diagnosis.csv
"""
import os
import csv
import glob
from collections import defaultdict

from PIL import Image

ROOT = os.path.join(os.environ["USERPROFILE"], "Desktop", "rice")
PRED_LBL = os.path.join(ROOT, "runs", "predict", "pred_val_v8n", "labels")
VAL_IMG = os.path.join(ROOT, "dataset_fieldlike", "images", "val")
VAL_LBL = os.path.join(ROOT, "dataset_fieldlike", "labels", "val")
OUT = os.path.join(ROOT, "checks", "pred_lowconf_audit")

BLIGHT = 3
CONF = 0.25       # standard reporting threshold
IOU_MIN = 0.5


def load(path, cls=None, conf_min=None):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if cls is not None and c != cls:
                continue
            cf = float(p[5]) if len(p) >= 6 else 1.0
            if conf_min is not None and cf < conf_min:
                continue
            out.append((float(p[1]), float(p[2]), float(p[3]), float(p[4]), cf))
    return out


def to_px(b, W, H):
    cx, cy, w, h = b[:4]
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


rows = []
for stem in sorted({os.path.splitext(os.path.basename(f))[0]
                    for f in glob.glob(os.path.join(PRED_LBL, "*.txt"))}):
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        c = os.path.join(VAL_IMG, stem + ext)
        if os.path.exists(c):
            img_path = c
            break
    if img_path is None:
        continue

    gt = load(os.path.join(VAL_LBL, stem + ".txt"), cls=BLIGHT)
    all_pred = load(os.path.join(PRED_LBL, stem + ".txt"), cls=BLIGHT)
    conf_pred = [b for b in all_pred if b[4] >= CONF]
    low_pred = [b for b in all_pred if b[4] < 0.05]

    # only diagnose images that actually have blight involvement
    if not gt and not conf_pred and not low_pred:
        continue

    W, H = Image.open(img_path).size
    gt_px = [to_px(b, W, H) for b in gt]
    pr_px = [(to_px(b, W, H), b[4]) for b in sorted(conf_pred, key=lambda x: -x[4])]

    matched_gt = set()
    tp = 0
    for box, _ in pr_px:
        best, best_j = 0.0, -1
        for j, g in enumerate(gt_px):
            if j in matched_gt:
                continue
            v = iou(box, g)
            if v > best:
                best, best_j = v, j
        if best >= IOU_MIN:
            matched_gt.add(best_j)
            tp += 1

    fp = len(pr_px) - tp
    fn = len(gt_px) - len(matched_gt)

    if not gt:
        continue  # keep the table focused on real blight images

    rows.append((stem, len(gt_px), tp, fp, fn, len(low_pred)))

rows.sort(key=lambda r: (-r[4], -r[5]))

path = os.path.join(OUT, "blight_image_diagnosis.csv")
with open(path, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["image", "n_gt", "TP", "FP", "FN", "n_pred_conf_lt0.05"])
    for r in rows:
        w.writerow(r)

tot_gt = sum(r[1] for r in rows)
tot_tp = sum(r[2] for r in rows)
tot_fp = sum(r[3] for r in rows)
tot_fn = sum(r[4] for r in rows)
worst = [r for r in rows if r[4] > 0]

print("blight images diagnosed :", len(rows))
print("GT boxes / TP / FP / FN : %d / %d / %d / %d" % (tot_gt, tot_tp, tot_fp, tot_fn))
print("recall @conf>=0.25,iou>=0.5 : %.3f" % (tot_tp / tot_gt if tot_gt else 0))
print("precision                    : %.3f" % (tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0))
print()
print("worst 15 images (by FN):")
for r in worst[:15]:
    print("  %-24s gt=%d TP=%d FP=%d FN=%d lowconf=%d" % r)
print()
print("images with FN>0 :", len(worst), "of", len(rows))
print("images clean (FN=0):", len(rows) - len(worst))
print("out:", path)
