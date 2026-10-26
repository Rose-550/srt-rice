"""Prediction audit for pred_val_v8n.

Outputs (all under checks/pred_lowconf_audit/):
  pred_manifest.csv          every prediction, sorted by confidence, with bucket
  bucket_summary.csv         per-class counts by confidence bucket
  blight_lowconf_ranking.csv blight images ranked by # of conf<0.05 boxes
  blight_lowconf_grid.png    top offenders, drawn with ONLY class-3 boxes
"""
import os
import csv
import glob
from collections import defaultdict, Counter

from PIL import Image, ImageDraw

ROOT = os.path.join(os.environ["USERPROFILE"], "Desktop", "rice")
PRED_LBL = os.path.join(ROOT, "runs", "predict", "pred_val_v8n", "labels")
VAL_IMG = os.path.join(ROOT, "dataset_fieldlike", "images", "val")
OUT = os.path.join(ROOT, "checks", "pred_lowconf_audit")
os.makedirs(OUT, exist_ok=True)

NAMES = {0: "稻纵卷叶螟", 1: "稻水象甲", 2: "稻叶蝉",
         3: "白叶枯病", 4: "胡麻斑病", 5: "稻瘟病"}
BLIGHT = 3
COLS, ROWS, CELL = 4, 3, 400


def bucket(c):
    if c < 0.05:
        return "lt0.05"
    if c < 0.25:
        return "0.05-0.25"
    if c < 0.50:
        return "0.25-0.50"
    return "ge0.50"


# ---- load predictions -------------------------------------------------
rows = []
for f in glob.glob(os.path.join(PRED_LBL, "*.txt")):
    stem = os.path.splitext(os.path.basename(f))[0]
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 6:
                continue
            rows.append((stem, int(float(p[0])), float(p[5]),
                         float(p[1]), float(p[2]), float(p[3]), float(p[4])))

# ---- manifest ---------------------------------------------------------
with open(os.path.join(OUT, "pred_manifest.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["image", "pred_class", "class_name", "conf", "conf_bucket",
                "cx", "cy", "w", "h"])
    for r in sorted(rows, key=lambda x: x[2]):
        w.writerow([r[0], r[1], NAMES[r[1]], "%.4f" % r[2], bucket(r[2]),
                    "%.6f" % r[3], "%.6f" % r[4], "%.6f" % r[5], "%.6f" % r[6]])

# ---- per-class bucket summary ----------------------------------------
with open(os.path.join(OUT, "bucket_summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["class_id", "class_name", "total", "lt0.05",
                "0.05-0.25", "0.25-0.50", "ge0.50"])
    for c in range(6):
        sub = [r for r in rows if r[1] == c]
        cnt = Counter(bucket(r[2]) for r in sub)
        w.writerow([c, NAMES[c], len(sub), cnt["lt0.05"],
                    cnt["0.05-0.25"], cnt["0.25-0.50"], cnt["ge0.50"]])

# ---- blight low-confidence ranking -----------------------------------
low = defaultdict(int)
high = defaultdict(int)
for stem, cls, conf, *_ in rows:
    if cls == BLIGHT:
        if conf < 0.05:
            low[stem] += 1
        elif conf >= 0.50:
            high[stem] += 1

rank = sorted(low.items(), key=lambda kv: (-kv[1], kv[0]))
with open(os.path.join(OUT, "blight_lowconf_ranking.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["image", "n_blight_conf_lt0.05", "n_blight_conf_ge0.50"])
    for stem, n in rank:
        w.writerow([stem, n, high.get(stem, 0)])

# ---- visual grid: only class-3 boxes ---------------------------------
picks = [s for s, _ in rank[:COLS * ROWS]]

# per-image blight boxes
boxes = defaultdict(list)
for stem, cls, conf, cx, cy, bw, bh in rows:
    if cls == BLIGHT:
        boxes[stem].append((conf, cx, cy, bw, bh))

sheet = Image.new("RGB", (COLS * CELL, ROWS * CELL), (25, 25, 25))
draw = ImageDraw.Draw(sheet)
placed = 0

for stem in picks:
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        cand = os.path.join(VAL_IMG, stem + ext)
        if os.path.exists(cand):
            img_path = cand
            break
    if img_path is None:
        continue

    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    scale = min(CELL / W, CELL / H)
    im = im.resize((max(1, int(W * scale)), max(1, int(H * scale))))

    tile = Image.new("RGB", (CELL, CELL), (25, 25, 25))
    tile.paste(im, ((CELL - im.width) // 2, (CELL - im.height) // 2))
    td = ImageDraw.Draw(tile)
    ox, oy = (CELL - im.width) // 2, (CELL - im.height) // 2

    n_low = n_high = 0
    for conf, cx, cy, bw, bh in boxes.get(stem, []):
        # normalized -> pixel in the resized tile
        x1 = ox + (cx - bw / 2) * im.width
        y1 = oy + (cy - bh / 2) * im.height
        x2 = ox + (cx + bw / 2) * im.width
        y2 = oy + (cy + bh / 2) * im.height
        if conf < 0.05:
            col, n_low = (255, 60, 60), n_low + 1          # red = junk
        elif conf >= 0.50:
            col, n_high = (60, 255, 60), n_high + 1        # green = confident
        else:
            col = (255, 200, 60)                           # amber = mid
        td.rectangle([x1, y1, x2, y2], outline=col, width=2)

    td.text((6, 6), "%s  low=%d high=%d" % (stem, n_low, n_high), fill=(255, 255, 255))

    gx, gy = (placed % COLS) * CELL, (placed // COLS) * CELL
    sheet.paste(tile, (gx, gy))
    placed += 1

sheet.save(os.path.join(OUT, "blight_lowconf_grid.png"))

print("predictions        :", len(rows))
print("blight low<0.05 img:", len(low), "images")
print("grid tiles placed  :", placed)
print("output dir         :", OUT)
