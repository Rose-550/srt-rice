# -*- coding: utf-8 -*-
"""Pass 4: provenance counting + residual-leak examples + summary report."""
import os, re, csv
import numpy as np

ROOT = os.path.abspath('.')
OUT = os.path.join(ROOT, 'checks', 'external_dedup')
CLASSES = ['BacterialLeafBlight','BrownSpot','Healthy','Hispa','LeafBlast','LeafScald',
           'LeafSmut','NarrowBrownLeafSpot','NeckBlast']
RF = re.compile(r'\.rf\.[0-9a-fA-F]+$'); EXT_TAG = re.compile(r'_(jpg|jpeg|png|JPG|JPEG|PNG)$')
OR = re.compile(r'_(orig|rotated)(?=_|$)')
def base(s): return EXT_TAG.sub('', RF.sub('', s))
def fam(s): return OR.sub('', base(s))

ze = np.load(os.path.join(OUT, 'ext_hash.npz'), allow_pickle=False)
e_name = ze['name']; e_stem = ze['stem']; e_cls = ze['cls']
zp = np.load(os.path.join(OUT, 'prj_hash.npz'), allow_pickle=False)
p_name = zp['name']; p_split = zp['split']
_selp = os.path.join(OUT, 'added_sel.npy')
blb0 = np.array([i for i, s in enumerate(e_cls) if s.split(',')[0] == '0'], dtype=int)
stems0 = e_stem[blb0].tolist()
_bd = os.path.join(ROOT, 'datasets', 'dataset_fieldlike_ext', 'images', 'train')
_built = set(re.sub(r'\.(jpg|jpeg|png|JPG|JPEG|PNG)$', '', n[len('extblb_'):]) for n in os.listdir(_bd) if n.startswith('extblb_'))
_pos = {s: i for i, s in enumerate(stems0)}
sel = np.array(sorted(_pos[s] for s in _built if s in _pos), dtype=int)
np.save(_selp, sel)
blb = np.array([i for i, s in enumerate(e_cls) if s.split(',')[0] == '0'], dtype=int)
stems = e_stem[blb].tolist()

def dist_matrix(HA, HB, variants=(0,1,2,3,4,5,6,7), chunk=20):
    nA, nB = len(HA), len(HB)
    D = np.full((nA, nB), 9999, dtype=np.int16)
    Bc = HB[:, 0, :]; Ac = HA[:, 0, :]
    idx = np.array(list(variants))
    for s in range(0, nA, chunk):
        e = min(s + chunk, nA)
        x = HA[s:e][:, idx][:, :, None, :] ^ Bc[None, None, :, :]
        D[s:e] = np.bitwise_count(x).sum(axis=-1).min(axis=1)
    for s in range(0, nB, chunk):
        e = min(s + chunk, nB)
        x = HB[s:e][:, idx][:, :, None, :] ^ Ac[None, None, :, :]
        d = np.bitwise_count(x).sum(axis=-1).min(axis=1)
        D[:, s:e] = np.minimum(D[:, s:e], d.T)
    return D

_dep = os.path.join(OUT, 'DEP.npy')
if os.path.isfile(_dep):
    DEP = np.load(_dep)
else:
    DEP = dist_matrix(ze['hashes'][blb], zp['hashes'])
    np.save(_dep, DEP)
    print('recomputed DEP and saved', flush=True)

lines = []
P = lambda s='': (print(s, flush=True), lines.append(s))

P('=== external dataset provenance (source-name based) ===')
P('%-24s %7s %9s %9s' % ('class', 'images', 'names', 'families'))
for c in range(9):
    idx = [i for i, s in enumerate(e_cls) if str(c) in s.split(',')]
    st = [e_stem[i] for i in idx]
    P('%-24s %7d %9d %9d' % (CLASSES[c], len(idx), len(set(base(s) for s in st)), len(set(fam(s) for s in st))))
P('%-24s %7d %9d %9d' % ('ALL', len(e_stem), len(set(base(s) for s in e_stem)), len(set(fam(s) for s in e_stem))))

P('')
P('=== the 1496 images actually merged into training ===')
a_base = [base(stems[i]) for i in sel]; a_fam = [fam(stems[i]) for i in sel]
P('files on disk            : 1496')
P('distinct source names    : %d' % len(set(a_base)))
P('distinct families (orig/rotated merged): %d' % len(set(a_fam)))
from collections import Counter
gc = Counter(a_base)
P('group-size distribution  : %s' % ', '.join('%dx:%d' % (k, v) for k, v in sorted(Counter(gc.values()).items())))
P('largest source groups    : %s' % ', '.join('%s(%d)' % (k, v) for k, v in gc.most_common(5)))

P('')
P('=== residual leakage of the ADDED images into project val (rot+mirror invariant) ===')
val_mask = (p_split == 'val')
Dsel = DEP[sel]
d_val = Dsel[:, val_mask].min(axis=1); d_trn = Dsel[:, ~val_mask].min(axis=1)
for t in [0, 2, 4, 8, 16]:
    P('  dist <= %-3d : to project val = %-4d   to project train = %-4d' % (t, int((d_val <= t).sum()), int((d_trn <= t).sum())))

P('')
P('=== added images within dist<=8 of a project VAL image (raw evidence) ===')
for i in np.nonzero(d_val <= 8)[0]:
    e = sel[i]
    js = np.nonzero(val_mask & (DEP[e] <= 8))[0]
    for j in js[:2]:
        P('  d=%-2d  ext: %s' % (DEP[e, j], e_name[blb[e]]))
        P('        val: %s' % p_name[j])

P('')
P('=== hash-identical (dist 0) external <-> project TRAIN examples ===')
for i in np.nonzero(d_trn == 0)[0]:
    e = sel[i]
    js = np.nonzero((~val_mask) & (DEP[e] == 0))[0]
    P('  ext: %s' % e_name[blb[e]])
    P('  train: %s' % ', '.join(p_name[js][:4]))

with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

with open(os.path.join(OUT, 'added_source_groups.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['source_name', 'n_files_in_added_set'])
    for k, v in sorted(gc.items()):
        w.writerow([k, v])

# emit concrete paths for visual verification
import json
pairs = []
for i in np.nonzero(d_val <= 4)[0][:3]:
    e = sel[i]
    j = int(np.nonzero(val_mask & (DEP[e] <= 4))[0][0])
    pairs.append({'d': int(DEP[e, j]), 'ext': e_name[blb[e]], 'prj': p_name[j], 'split': 'val'})
for i in np.nonzero(d_trn == 0)[0][:2]:
    e = sel[i]
    j = int(np.nonzero((~val_mask) & (DEP[e] == 0))[0][0])
    pairs.append({'d': 0, 'ext': e_name[blb[e]], 'prj': p_name[j], 'split': 'train'})
with open(os.path.join(OUT, 'visual_check.json'), 'w', encoding='utf-8') as f:
    json.dump(pairs, f, ensure_ascii=False, indent=1)
P('')
P('visual_check pairs: %s' % json.dumps(pairs, ensure_ascii=False))
