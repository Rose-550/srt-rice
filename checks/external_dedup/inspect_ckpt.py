# -*- coding: utf-8 -*-
import os, glob
ROOT = os.path.abspath('.')
names_en = [b'rice_leaf_roller', b'rice_water_weevil', b'rice_leafhopper', b'bacterial_blight', b'brown_spot', b'rice_blast']
names_cn = ['稻纵卷叶螟'.encode('utf-8'), '稻水象甲'.encode('utf-8'), '稻叶蝉'.encode('utf-8'),
            '白叶枯病'.encode('utf-8'), '胡麻斑病'.encode('utf-8'), '稻瘟病'.encode('utf-8')]
cands = []
for p in ['best.pt', 'best_old.pt', 'best_names_fixed.pt', 'yolov8n.pt', 'yolo26n.pt']:
    if os.path.isfile(p): cands.append(p)
for p in sorted(glob.glob(os.path.join('runs', 'detect', '*', 'weights', 'best.pt'))):
    cands.append(p)
print('%-58s %8s %-14s %-14s' % ('checkpoint', 'MB', 'english names', 'chinese names'))
for p in cands:
    b = open(p, 'rb').read()
    ne = sum(1 for n in names_en if n in b)
    nc = sum(1 for n in names_cn if n in b)
    print('%-58s %8.1f %-14s %-14s' % (p, len(b)/1e6, '%d/6' % ne, '%d/6' % nc))
