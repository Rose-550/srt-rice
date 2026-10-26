"""Collect the 5-fold CV results into a tidy CSV + mean/std summary.

Parses the two console logs. Each fold ends with a validation table of the form

                       all        935       1248      0.923       0.86      0.904      0.554
                             308        394      ...
                     (6 class rows follow, class names print blank on this machine)

so a table is detected as an "all" row followed by 6 numeric rows.

Output: checks/cv5/cv5_summary.csv, checks/cv5/cv5_mean_std.csv
"""
import os
import re
import csv

ROOT = os.path.join(os.environ["USERPROFILE"], "Desktop", "rice")
LOGS = [("fold0,fold1", os.path.join(ROOT, "train_cv5.console.log")),
        ("fold2,fold3,fold4", os.path.join(ROOT, "train_cv5_folds234.console.log"))]
OUT = os.path.join(ROOT, "checks", "cv5")
os.makedirs(OUT, exist_ok=True)

NAMES = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉", "白叶枯病", "胡麻斑病", "稻瘟病"]
ALL_RE = re.compile(r"^\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
CLS_RE = re.compile(r"^\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def tables(path):
    """Yield (all_row, [6 class rows]) for every final table in the log."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    lines = raw.replace("\r", "\n").split("\n")
    out = []
    i = 0
    while i < len(lines):
        m = ALL_RE.match(lines[i])
        if m:
            blk = []
            j = i + 1
            while j < len(lines) and len(blk) < 6:
                cm = CLS_RE.match(lines[j])
                if not cm:
                    break
                blk.append(cm.groups())
                j += 1
            if len(blk) == 6:
                out.append((m.groups(), blk))
                i = j
                continue
        i += 1
    return out


rows = []
fold = 0
for label, path in LOGS:
    if not os.path.exists(path):
        print("missing log:", path)
        continue
    for all_row, cls_rows in tables(path):
        rows.append((fold, "all", "all", all_row))
        for k, cr in enumerate(cls_rows):
            rows.append((fold, k, NAMES[k], cr))
        fold += 1

with open(os.path.join(OUT, "cv5_summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["fold", "class_id", "class_name", "Images", "Instances",
                "Precision", "Recall", "mAP50", "mAP50-95"])
    for f, cid, cname, r in rows:
        w.writerow([f, cid, cname] + list(r[:2]) + ["%.4f" % float(x) for x in r[2:]])

# ---- mean / std across folds -----------------------------------------
groups = {}
for f, cid, cname, r in rows:
    groups.setdefault((cid, cname), []).append([float(x) for x in r[2:]])


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


order = [("all", "all")] + [(i, NAMES[i]) for i in range(6)]
with open(os.path.join(OUT, "cv5_mean_std.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["class_id", "class_name", "n_folds", "P_mean", "P_std", "R_mean", "R_std",
                "mAP50_mean", "mAP50_std", "mAP50-95_mean", "mAP50-95_std",
                "mAP50_min", "mAP50_max", "mAP50-95_min", "mAP50-95_max"])
    for key in order:
        if key not in groups:
            continue
        arr = groups[key]
        pm, ps = stats([a[0] for a in arr])
        rm, rs = stats([a[1] for a in arr])
        m5, s5 = stats([a[2] for a in arr])
        m9, s9 = stats([a[3] for a in arr])
        cid, cname = key
        w.writerow([cid, cname, len(arr),
                    "%.4f" % pm, "%.4f" % ps, "%.4f" % rm, "%.4f" % rs,
                    "%.4f" % m5, "%.4f" % s5, "%.4f" % m9, "%.4f" % s9,
                    "%.4f" % min(a[2] for a in arr), "%.4f" % max(a[2] for a in arr),
                    "%.4f" % min(a[3] for a in arr), "%.4f" % max(a[3] for a in arr)])

# ---- report -----------------------------------------------------------
print("folds parsed:", fold)
print()
print("%-12s %18s %18s" % ("class", "mAP50 (mean±std)", "mAP50-95 (mean±std)"))
for key in order:
    if key not in groups:
        continue
    arr = groups[key]
    m5, s5 = stats([a[2] for a in arr])
    m9, s9 = stats([a[3] for a in arr])
    print("%-12s   %7.4f ± %.4f    %7.4f ± %.4f" % (key[1], m5, s5, m9, s9))
print()
print("per-fold mAP50-95 spread (all):")
arr = groups[("all", "all")]
for i, a in enumerate(arr):
    print("  fold %d : mAP50 %.4f   mAP50-95 %.4f" % (i, a[2], a[3]))
print()
print("out:", OUT)
