# -*- coding: utf-8 -*-
import os
import numpy as np
ROOT = os.path.abspath('.'); OUT = os.path.join(ROOT, 'checks', 'external_dedup')
ze = np.load(os.path.join(OUT, 'ext_hash.npz'), allow_pickle=False)
zp = np.load(os.path.join(OUT, 'prj_hash.npz'), allow_pickle=False)
e_name, e_cls = ze['name'], ze['cls']
p_name, p_split = zp['name'], zp['split']
DEP = np.load(os.path.join(OUT, 'DEP.npy')); sel = np.load(os.path.join(OUT, 'added_sel.npy'))
blb = np.array([i for i, s in enumerate(e_cls) if s.split(',')[0] == '0'], dtype=int)
val_mask = (p_split == 'val')
# project BLB = filenames starting with bacterial_blight
p_blb = np.array([n.lower().startswith('bacterial_blight') for n in p_name])
val_blb = val_mask & p_blb
train_blb = (~val_mask) & p_blb
Dsel = DEP[sel]
print('project val BLB images: %d ; project train BLB images: %d' % (val_blb.sum(), train_blb.sum()))
print('')
print('%-10s %18s %22s %22s' % ('dist<=', 'added involved', 'distinct VAL BLB hit', 'distinct TRAIN BLB hit'))
for t in [0, 2, 4, 8, 16]:
    ev = np.nonzero(Dsel[:, val_blb].min(axis=1) <= t)[0]
    vals = set(); trs = set()
    for i in ev:
        for j in np.nonzero(val_blb & (DEP[sel[i]] <= t))[0]:
            vals.add(int(j))
        for j in np.nonzero(train_blb & (DEP[sel[i]] <= t))[0]:
            trs.add(int(j))
    print('%-10d %18d %16d (%4.1f%%) %16d (%4.1f%%)' % (t, len(ev), len(vals), 100.0*len(vals)/val_blb.sum(), len(trs), 100.0*len(trs)/max(train_blb.sum(),1)))
print('')
print('val BLB images hit at dist<=8:')
ev = np.nonzero(Dsel[:, val_blb].min(axis=1) <= 8)[0]
vals = {}
for i in ev:
    for j in np.nonzero(val_blb & (DEP[sel[i]] <= 8))[0]:
        vals.setdefault(p_name[j], []).append('%s(d=%d)' % (e_name[blb[sel[i]]], DEP[sel[i], j]))
for k in sorted(vals):
    print('  %-28s <- %s' % (k, ', '.join(vals[k][:2])))
