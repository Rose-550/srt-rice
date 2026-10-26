# -*- coding: utf-8 -*-
"""Pass 5: how many project val images are contaminated + consolidated summary."""
import os, re
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
out = []
P = lambda s='': (print(s, flush=True), out.append(s))
P('=== contamination of project val caused by the %d added images ===' % len(sel))
P('val BLB images total: %d' % int(val_mask.sum()))
for t in [2, 4, 8, 16]:
    ev = np.nonzero(Dsel[:, val_mask].min(axis=1) <= t)[0]
    vals = set()
    for i in ev:
        for j in np.nonzero(val_mask & (DEP[sel[i]] <= t))[0]:
            vals.add(int(j))
    P('  dist<=%-3d : added images involved = %-4d | distinct VAL images hit = %-3d (%.0f%% of val BLB)'
      % (t, len(ev), len(vals), 100.0 * len(vals) / int(val_mask.sum())))
P('')
P('=== same-source-name grouping, per external class ===')
RF = re.compile(r'\.rf\.[0-9a-fA-F]+$'); TAG = re.compile(r'_(jpg|jpeg|png|JPG|JPEG|PNG)$')
OR = re.compile(r'_(orig|rotated)(?=_|$)')
b = lambda s: TAG.sub('', RF.sub('', s)); f = lambda s: OR.sub('', b(s))
CLS = ['BacterialLeafBlight','BrownSpot','Healthy','Hispa','LeafBlast','LeafScald','LeafSmut','NarrowBrownLeafSpot','NeckBlast']
P('%-24s %7s %9s %9s %7s' % ('class','images','sources','families','inflation'))
for c in range(9):
    st = [e_name[i] for i, s in enumerate(e_cls) if str(c) in s.split(',')]
    st = [os.path.splitext(x)[0] for x in st]
    n = len(st); src = len(set(b(s) for s in st)); fm = len(set(f(s) for s in st))
    P('%-24s %7d %9d %9d %6.2fx' % (CLS[c], n, src, fm, n / max(src, 1)))
st = [os.path.splitext(x)[0] for x in e_name]
P('%-24s %7d %9d %9d %6.2fx' % ('ALL', len(st), len(set(b(s) for s in st)), len(set(f(s) for s in st)), len(st)/len(set(b(s) for s in st))))
with open(os.path.join(OUT, 'contamination_summary.txt'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(out) + '\n')
P('')
P('written -> checks/external_dedup/contamination_summary.txt')
