# -*- coding: utf-8 -*-
import os, csv
import numpy as np
ROOT = os.path.abspath('.'); OUT = os.path.join(ROOT, 'checks', 'external_dedup')
ze = np.load(os.path.join(OUT, 'ext_hash.npz'), allow_pickle=False)
zp = np.load(os.path.join(OUT, 'prj_hash.npz'), allow_pickle=False)
e_name, e_cls = ze['name'], ze['cls']
p_name, p_split = zp['name'], zp['split']
DEP = np.load(os.path.join(OUT, 'DEP.npy')); sel = np.load(os.path.join(OUT, 'added_sel.npy'))
blb = np.array([i for i, s in enumerate(e_cls) if s.split(',')[0] == '0'], dtype=int)
val_mask = (p_split == 'val')
Dsel = DEP[sel]
d_val = Dsel[:, val_mask].min(axis=1); d_trn = Dsel[:, ~val_mask].min(axis=1)
with open(os.path.join(OUT, 'added_vs_project_val.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['ext_name', 'min_dist_to_project_val', 'min_dist_to_project_train'])
    for i in range(len(sel)):
        w.writerow([e_name[blb[sel[i]]], int(d_val[i]), int(d_trn[i])])
print('rows written:', len(sel))
print('sanity: <=4 val=%d train=%d | <=8 val=%d train=%d'
      % ((d_val <= 4).sum(), (d_trn <= 4).sum(), (d_val <= 8).sum(), (d_trn <= 8).sum()))
