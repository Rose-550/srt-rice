# -*- coding: utf-8 -*-
"""Pass 3: contamination check - are the ADDED external images near-dups of project val?"""
import os, re, csv, time
import numpy as np

ROOT = os.path.abspath('.')
OUT = os.path.join(ROOT, 'checks', 'external_dedup')
FULL8 = (0,1,2,3,4,5,6,7)

def dist_matrix(HA, HB, variants=FULL8, chunk=20):
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

ze = np.load(os.path.join(OUT, 'ext_hash.npz'), allow_pickle=False)
he = ze['hashes']; e_name = ze['name']; e_stem = ze['stem']; e_cls = ze['cls']
zp = np.load(os.path.join(OUT, 'prj_hash.npz'), allow_pickle=False)
hp = zp['hashes']; p_name = zp['name']; p_split = zp['split']

blb = np.array([i for i, s in enumerate(e_cls) if s.split(',')[0] == '0'], dtype=int)
HB = he[blb]

built_dir = os.path.join(ROOT, 'dataset_fieldlike_ext', 'images', 'train')
built = [n for n in os.listdir(built_dir) if n.startswith('extblb_')]
built_src = set(re.sub(r'\.(jpg|jpeg|png|JPG|JPEG|PNG)$', '', n[len('extblb_'):]) for n in built)
stems = e_stem[blb].tolist()
pos = {s: i for i, s in enumerate(stems)}
sel = np.array(sorted(pos[s] for s in built_src if s in pos), dtype=int)
print('added files on disk: %d | matched in ext table: %d' % (len(built), len(sel)))

# ---- provenance counting ----
RF = re.compile(r'\.rf\.[0-9a-fA-F]+$'); EXT_TAG = re.compile(r'_(jpg|jpeg|png|JPG|JPEG|PNG)$')
OR = re.compile(r'_(orig|rotated)(?=_|$)')
def base(s): return EXT_TAG.sub('', RF.sub('', s))
def fam(s): return OR.sub('', base(s))
a_base = [base(stems[i]) for i in sel]; a_fam = [fam(stems[i]) for i in sel]
print('added subse  : unique source names=%d | merging orig/rotated=%d' % (len(set(a_base)), len(set(a_fam))))
all_base = [base(s) for s in stems]; all_fam = [fam(s) for s in stems]
print('full BLB 2000: unique source names=%d | merging orig/rotated=%d' % (len(set(all_base)), len(set(all_fam))))

# ---- distance from every added image to each project split ----
t0 = time.time()
DEP = dist_matrix(HB, hp)
print('dist computed in %.1fs' % (time.time() - t0))
val_mask = (p_split == 'val')
Dsel = DEP[sel]
d_val = Dsel[:, val_mask].min(axis=1)
d_trn = Dsel[:, ~val_mask].min(axis=1)
print('\n=== of the %d ADDED images: min distance to ANY project image ===' % len(sel))
for t in [0, 2, 4, 8, 16]:
    print('  <= %-3d  vs project val: %-5d   vs project train: %-5d' % (t, int((d_val <= t).sum()), int((d_trn <= t).sum())))

print('\nadded images that are exact-hash (dist 0) matches of a PROJECT VAL image:')
hits = np.nonzero(d_val == 0)[0]
for i in hits[:20]:
    e = blb[sel[i]]
    j = np.nonzero(val_mask & (DEP[e] == 0))[0]
    print('  %s  <->  %s' % (e_name[e], ', '.join(p_name[j][:4])))
if len(hits) == 0:
    print('  (none)')

print('\nadded images with dist<=4 to a project val image:')
hits4 = np.nonzero(d_val <= 4)[0]
print('  count = %d' % len(hits4))
for i in hits4[:10]:
    e = blb[sel[i]]
    js = np.nonzero(val_mask & (DEP[e] <= 4))[0]
    print('  d=%d %s <-> %s' % (DEP[e, js[0]], e_name[e], ', '.join(p_name[js][:3])))

with open(os.path.join(OUT, 'added_vs_project_val.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['ext_name', 'min_dist_to_val', 'min_dist_to_train'])
    for i in range(len(sel)):
        w.writerow([e_name[blb[sel[i]]], int(d_val[i]), int(d_trn[i])])

np.save(os.path.join(OUT, 'added_sel.npy'), sel)
np.save(os.path.join(OUT, 'DEP.npy'), DEP)
print('\nsaved added_vs_project_val.csv')
