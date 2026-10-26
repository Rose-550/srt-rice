"""Re-validate each fold's best.pt to get FULL-PRECISION per-class metrics.

The console summary table only prints 3 decimals, which quantises the CV standard
deviations. Reading metrics.box arrays gives the exact floats instead.

Output: checks/cv5/cv5_fullprec_perfold.csv, checks/cv5/cv5_fullprec_mean_std.csv
"""
import os
import csv

import numpy as np
from ultralytics import YOLO

ROOT = os.path.join(os.environ["USERPROFILE"], "Desktop", "rice")
CV = os.path.join(os.environ["USERPROFILE"], ".trae-cn", "work",
                  "6aa66b038e42140e46b27278", "cv_splits")
OUT = os.path.join(ROOT, "checks", "cv5")
os.makedirs(OUT, exist_ok=True)

NAMES = {0: "稻纵卷叶螟", 1: "稻水象甲", 2: "稻叶蝉",
         3: "白叶枯病", 4: "胡麻斑病", 5: "稻瘟病"}

per_fold = {}   # fold -> {class_id: (P, R, mAP50, mAP50-95)}

for f in range(5):
    best = os.path.join(ROOT, "runs", "detect", f"cv_fold_{f}", "weights", "best.pt")
    yaml = os.path.join(CV, f"cv_fold_{f}.yaml")
    if not (os.path.exists(best) and os.path.exists(yaml)):
        print("skip fold %d (missing %s)" % (f, "weights" if not os.path.exists(best) else "yaml"))
        continue

    model = YOLO(best)
    r = model.val(data=yaml, split="val", imgsz=640, batch=16, device=0,
                  conf=0.001, iou=0.7, plots=False, verbose=False, workers=0,
                  project=OUT, name=f"val_fold{f}", exist_ok=True)
    box = r.box
    idx = np.asarray(box.ap_class_index).astype(int)
    d = {}
    for j, c in enumerate(idx.tolist()):
        d[c] = (float(box.p[j]), float(box.r[j]),
                float(box.ap50[j]), float(box.ap[j]))
    per_fold[f] = d
    print("fold %d val done. mp=%.4f mr=%.4f map50=%.4f map=%.4f"
          % (f, box.mp, box.mr, box.map50, box.map), flush=True)

# ---- per-fold CSV -----------------------------------------------------
with open(os.path.join(OUT, "cv5_fullprec_perfold.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["fold", "class_id", "class_name", "Precision", "Recall", "mAP50", "mAP50-95"])
    for f in sorted(per_fold):
        for c in sorted(per_fold[f]):
            p, rr, m5, m9 = per_fold[f][c]
            w.writerow([f, c, NAMES[c], "%.6f" % p, "%.6f" % rr, "%.6f" % m5, "%.6f" % m9])

# ---- mean / std -------------------------------------------------------
folds = sorted(per_fold)
classes = sorted({c for f in folds for c in per_fold[f]})


def ms(v):
    a = np.asarray(v, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


with open(os.path.join(OUT, "cv5_fullprec_mean_std.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["class_id", "class_name", "n", "P_mean", "P_std", "R_mean", "R_std",
                "mAP50_mean", "mAP50_std", "mAP50-95_mean", "mAP50-95_std",
                "mAP50_min", "mAP50_max", "mAP50-95_min", "mAP50-95_max"])
    summary = {}
    for c in classes:
        vals = [per_fold[f][c] for f in folds if c in per_fold[f]]
        p, rr, m5, m9 = zip(*vals)
        row = [c, NAMES[c], len(vals)]
        for series in (p, rr, m5, m9):
            m, s = ms(series)
            row += ["%.6f" % m, "%.6f" % s]
        row += ["%.6f" % min(m5), "%.6f" % max(m5), "%.6f" % min(m9), "%.6f" % max(m9)]
        w.writerow(row)
        summary[c] = (ms(m5), ms(m9))

print()
print("%-12s %21s %21s" % ("class", "mAP50 (mean ± std)", "mAP50-95 (mean ± std)"))
for c in classes:
    (m5, s5), (m9, s9) = summary[c]
    print("%-12s   %8.4f ± %.4f     %8.4f ± %.4f" % (NAMES[c], m5, s5, m9, s9))

print()
print("per-fold, per-class mAP50-95:")
hdr = "%-12s" % "class" + "".join("  fold%d " % f for f in folds)
print(hdr)
for c in classes:
    line = "%-12s" % NAMES[c]
    for f in folds:
        line += "  %.4f" % per_fold[f][c][3] if c in per_fold[f] else "   --   "
    print(line)

print()
print("out:", OUT)
