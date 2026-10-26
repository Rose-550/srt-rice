# -*- coding: utf-8 -*-
"""annotate_image 标注视觉预览：把新样式的检测框渲染成图，人工过目。

场景覆盖：常态位置 / 顶边 / 右边 / 同图多框（虫+病）/ 摄像头小帧。
产物：checks/_annotate_out/*.png
用法（rice-gpu 环境）：python checks/annotate_preview.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "_annotate_out"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["RICE_APP_DIR"] = str(OUT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import rice_dp1_fluent as m

IMG = ROOT / "data" / "raw" / "JPEGImages"


def render(src, specs, out_name, max_w=900):
    """specs: [(fx1, fy1, fx2, fy2, name, conf)] 用画面比例定位，适配任意图幅"""
    if isinstance(src, np.ndarray):
        img = src.copy()
    else:
        img = cv2.imread(str(IMG / src))
        assert img is not None, src
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s)
    h, w = img.shape[:2]
    boxes = [{"box": (int(fx1 * w), int(fy1 * h), int(fx2 * w), int(fy2 * h)),
              "name": name, "conf": conf} for fx1, fy1, fx2, fy2, name, conf in specs]
    out = m.annotate_image(img, boxes)
    ok = cv2.imwrite(str(OUT / out_name), out)
    fs = int(max(24, min(48, min(h, w) * 0.055)))
    lw = max(2, min(8, round(min(h, w) * 0.004)))
    print("%-28s %dx%d  字号 %d  线宽 %d  %s" % (out_name, w, h, fs, lw, "OK" if ok else "写盘失败"))


print("== annotate_image 预览（输出到 checks/_annotate_out/）==")

# 1. 常态位置 · 病害框（白叶枯病）
render("bacterial_blight_10.jpg",
       [(0.12, 0.30, 0.72, 0.78, "白叶枯病", 0.912)],
       "01_病害_常态.png")

# 2. 顶边框：标签放不下，应移进框内顶部
render("rice_blast_3.jpg",
       [(0.15, 0.01, 0.85, 0.45, "稻瘟病", 0.83)],
       "02_病害_顶边.png")

# 3. 右边框：标签应被夹回画面内
render("rice_leaf_roller_5.jpg",
       [(0.60, 0.35, 0.99, 0.80, "稻纵卷叶螟", 0.87)],
       "03_虫害_右边.png")

# 4. 同图双框：虫（金）+ 病（红）同时出现
render("rice_water_weevil_7.jpg",
       [(0.05, 0.10, 0.48, 0.55, "稻水象甲", 0.92),
        (0.52, 0.45, 0.95, 0.85, "胡麻斑病", 0.76)],
       "04_双框_虫加病.png")

# 5. 摄像头小帧（合成绿色田面 480x360）：小尺寸下的字号/线宽
frame = np.zeros((360, 480, 3), np.uint8)
frame[:, :] = (48, 96, 40)
frame[60:300, 100:400] = (60, 120, 52)
noise = np.random.default_rng(7).integers(0, 22, frame.shape, dtype=np.uint8)
frame = cv2.add(frame, noise)
render(frame,
       [(0.22, 0.20, 0.80, 0.75, "稻叶蝉", 0.66)],
       "05_摄像头小帧.png")

print("完成。")
