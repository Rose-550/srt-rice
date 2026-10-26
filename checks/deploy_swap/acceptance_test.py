# -*- coding: utf-8 -*-
"""Acceptance test: replicate exactly what rice_dp1.py does with the deployed best.pt."""
import os, sys
from ultralytics import YOLO

ROOT = os.path.abspath('.')
CLASS_NAMES = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉", "白叶枯病", "胡麻斑病", "稻瘟病"]   # rice_dp1.py line 67
import re
_src = open(os.path.join(ROOT, 'rice_dp1.py'), encoding='utf-8').read()
CONF_THRESHOLD = float(re.search(r'^CONF_THRESHOLD\s*=\s*([0-9.]+)', _src, re.M).group(1))
print('read CONF_THRESHOLD from rice_dp1.py =', CONF_THRESHOLD)

mp = os.path.join(ROOT, 'best.pt')
print('MODEL_PATH =', mp, '| exists:', os.path.exists(mp))
print('md5-ish size =', os.path.getsize(mp))

model = YOLO(mp)
print('ckpt model.names =', dict(model.names))
ok = (len(model.names) == 6) and all(model.names[i] == CLASS_NAMES[i] for i in range(6))
print('names order identical to app CLASS_NAMES:', ok)

VAL = os.path.join(ROOT, 'datasets', 'dataset_fieldlike', 'images', 'val')
prefixes = ['rice_leaf_roller', 'rice_water_weevil', 'rice_leafhopper',
            'bacterial_blight', 'brown_spot', 'rice_blast']
picks = []
for pre in prefixes:
    for n in sorted(os.listdir(VAL)):
        if n.startswith(pre) and n.lower().endswith('.jpg'):
            picks.append((pre, os.path.join(VAL, n)))
            break

print('')
for conf in (CONF_THRESHOLD, 0.6):
    print('--- conf=%.2f (%s) ---' % (conf, 'CURRENT app setting' if conf == CONF_THRESHOLD else 'old setting, for comparison'))
    for pre, p in picks:
        res = model(p, conf=conf, imgsz=640, verbose=False)[0]
        dets = ['%s %.3f' % (CLASS_NAMES[int(b.cls.item())], float(b.conf.item())) for b in res.boxes]
        print('  %-24s truth=%-16s -> %s' % (os.path.basename(p), pre, ', '.join(dets) if dets else '(no detection)'))
    print('')
