# -*- coding: utf-8 -*-
"""Pass 2: rotation+mirror-invariant near-duplicate clustering (read-only w.r.t. datasets)."""
import os, sys, time, csv, re
import numpy as np

ROOT = os.path.abspath('.')
OUT = os.path.join(ROOT, 'checks', 'external_dedup')
CLASSES = ['BacterialLeafBlight','BrownSpot','Healthy','Hispa','LeafBlast','LeafScald',
           'LeafSmut','NarrowBrownLeafSpot','NeckBlast']
THRS = [2, 4, 6, 8, 10, 12, 16, 20]
FULL8 = (0,1,2,3,4,5,6,7)
MIRROR_ONLY = (0, 4)

RF = re.compile(r'\.rf\.[0-9a-fA-F]+$')
EXT_TAG = re.compile(r'_(jpg|jpeg|png|JPG|JPEG|PNG|bmp|webp)$')
ORIGROT = re.compile(r'_(orig|rotated)(?=_|$)')

def base_stem(stem):
    s = RF.sub('', stem)
    s = EXT_TAG.sub('', s)
    return s

def family(stem):
    return ORIGROT.sub('', base_stem(stem))

def dist_matrix(HA, HB, variants=FULL8, chunk=25, verbose=''):
    nA, nB = len(HA), len(HB)
    D = np.full((nA, nB), 9999, dtype=np.int16)
    Bc = HB[:, 0, :]; Ac = HA[:, 0, :]
    idx = np.array(list(variants))
    t0 = time.time()
    for s in range(0, nA, chunk):
        e = min(s + chunk, nA)
        A = HA[s:e][:, idx]                      # (m, K, 4)
        x = A[:, :, None, :] ^ Bc[None, None, :, :]
        D[s:e] = np.bitwise_count(x).sum(axis=-1).min(axis=1)
    for s in range(0, nB, chunk):
        e = min(s + chunk, nB)
        B = HB[s:e][:, idx]
        x = B[:, :, None, :] ^ Ac[None, None, :, :]
        d = np.bitwise_count(x).sum(axis=-1).min(axis=1)
        D[:, s:e] = np.minimum(D[:, s:e], d.T)
    if verbose:
        print('    dist %s %dx%d in %.1fs' % (verbose, nA, nB, time.time() - t0), flush=True)
    return D

class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]; a = p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra

def cluster_labels(D, thr):
    n = D.shape[0]
    uf = UF(n)
    for i in range(n - 1):
        js = np.nonzero(D[i, i+1:] <= thr)[0]
        for j in js:
            uf.union(i, int(j) + i + 1)
    return np.array([uf.find(i) for i in range(n)])

def report(tag, D, thrs=THRS):
    rows = []
    n = D.shape[0]
    for t in thrs:
        lab = cluster_labels(D, t)
        _, cnt = np.unique(lab, return_counts=True)
        nc = len(cnt); sing = int((cnt == 1).sum()); mx = int(cnt.max())
        rows.append((tag, n, t, nc, sing, mx))
        print('  %-30s THR=%-3d clusters=%-6d singleton=%-6d max=%-3d' % (tag, t, nc, sing, mx), flush=True)
    return rows

all_rows = []
t_start = time.time()

ze = np.load(os.path.join(OUT, 'ext_hash.npz'), allow_pickle=False)
he = ze['hashes']; e_split = ze['split']; e_name = ze['name']; e_stem = ze['stem']; e_cls = ze['cls']
print('external total: %d' % len(he), flush=True)

def cls_of(s):
    try: return [int(x) for x in s.split(',')]
    except Exception: return []

by_class = {c: [] for c in range(9)}
for i, s in enumerate(e_cls):
    for c in cls_of(s):
        if 0 <= c < 9: by_class[c].append(i)
print('\n=== external per-class image counts (by label) ===')
for c in range(9):
    print('  %d %-22s %d' % (c, CLASSES[c], len(by_class[c])), flush=True)

# ---------- PASS A: BLB ----------
print('\n=== PASS A: BacterialLeafBlight (the class used by the ext experiment) ===')
blb = np.array(by_class[0], dtype=int)
HB = he[blb]; SB = e_stem[blb]
bases = [base_stem(s) for s in SB.tolist()]
fams = [family(s) for s in SB.tolist()]
n_uniq = len(set(SB.tolist())); n_base = len(set(bases)); n_fam = len(set(fams))
print('  images=%d | unique filenames=%d | unique base stems=%d | unique source families=%d'
      % (len(blb), n_uniq, n_base, n_fam), flush=True)
D8 = dist_matrix(HB, HB, FULL8, verbose='BLB full8')
print('\n  -- rotation+mirror invariant (NEW) --')
all_rows += report('external BLB rot+mirror', D8)
DM = dist_matrix(HB, HB, MIRROR_ONLY, verbose='BLB mirror-only')
print('\n  -- mirror only, no rotation (OLD build method) --')
all_rows += report('external BLB mirror-only', DM)
np.save(os.path.join(OUT, 'D_blb_full8.npy'), D8)
np.save(os.path.join(OUT, 'blb_index.npy'), blb)

# per-image cluster assignment CSV at THR 4 and 8
with open(os.path.join(OUT, 'blb_clusters.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['name', 'base_stem', 'family', 'cluster_thr4', 'cluster_thr8', 'cluster_thr16'])
    l4 = cluster_labels(D8, 4); l8 = cluster_labels(D8, 8); l16 = cluster_labels(D8, 16)
    for i in range(len(blb)):
        w.writerow([e_name[blb[i]], bases[i], fams[i], int(l4[i]), int(l8[i]), int(l16[i])])

# ---------- PASS B: the images actually added ----------
print('\n=== PASS B: images actually merged into training ===')
built_dir = os.path.join(ROOT, 'dataset_fieldlike_ext', 'images', 'train')
built = [n for n in os.listdir(built_dir) if n.startswith('extblb_')]
built_src = set(re.sub(r'\.(jpg|jpeg|png|JPG|JPEG|PNG)$', '', n[len('extblb_'):]) for n in built)
pos = {s: i for i, s in enumerate(SB.tolist())}
sel = np.array(sorted(pos[s] for s in built_src if s in pos), dtype=int)
missing = [s for s in built_src if s not in pos]
print('  built files=%d | matched %d | unmatched %d' % (len(built), len(sel), len(missing)), flush=True)
if missing: print('  unmatched sample: %s' % missing[:3], flush=True)
sub_base = [bases[i] for i in sel]; sub_fam = [fams[i] for i in sel]
print('  within added subset: unique base stems=%d | unique families=%d'
      % (len(set(sub_base)), len(set(sub_fam))), flush=True)
DS = D8[np.ix_(sel, sel)]
print('\n  -- rotation+mirror invariant (NEW) --')
all_rows += report('added rot+mirror', DS)
print('\n  -- mirror only, no rotation (OLD) --')
all_rows += report('added mirror-only', dist_matrix(HB[sel], HB[sel], MIRROR_ONLY))

# ---------- PASS C: every class ----------
print('\n=== PASS C: every external class (rot+mirror, all 2000-class images) ===')
for c in range(9):
    idx = np.array(by_class[c], dtype=int)
    if len(idx) < 2: continue
    Dc = dist_matrix(he[idx], he[idx])
    all_rows += report('%d %s' % (c, CLASSES[c]), Dc)
    print('', flush=True)

# ---------- PASS D: whole external ----------
print('\n=== PASS D: whole external dataset (%d) ===' % len(he))
try:
    D_all = dist_matrix(he, he, chunk=20, verbose='whole external')
    all_rows += report('external ALL rot+mirror', D_all, thrs=[4, 8, 16])
except Exception as e:
    print('  PASS D failed: %s' % e, flush=True)

# ---------- PASS E: external BLB vs project ----------
print('\n=== PASS E: external BLB (2000) vs project dataset_fieldlike (8396) ===')
zp = np.load(os.path.join(OUT, 'prj_hash.npz'), allow_pickle=False)
hp = zp['hashes']; p_split = zp['split']; p_name = zp['name']
print('  project images=%d (train=%d val=%d)'
      % (len(hp), int((p_split == 'train').sum()), int((p_split == 'val').sum())), flush=True)
DEP = dist_matrix(HB, hp, FULL8, chunk=20, verbose='BLB vs project')
rows_ep = []
for t in [0, 2, 4, 8, 16, 32]:
    ia, ip = np.nonzero(DEP <= t)
    n_val = int((p_split[ip] == 'val').sum())
    rows_ep.append((t, len(ia), int((p_split[ip] == 'train').sum()), n_val))
    print('  THR=%-3d pairs=%-7d project_side: train=%-7d val=%-5d' % (t, len(ia), rows_ep[-1][2], n_val), flush=True)
with open(os.path.join(OUT, 'ext_vs_project_pairs.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['dist', 'ext_name', 'project_name', 'project_split'])
    ia, ip = np.nonzero(DEP <= 16)
    for k in np.argsort(DEP[ia, ip]):
        w.writerow([int(DEP[ia[k], ip[k]]), e_name[blb[ia[k]]], p_name[ip[k]], p_split[ip[k]]])
with open(os.path.join(OUT, 'ext_vs_project_summary.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['thr', 'pairs', 'project_train', 'project_val']); w.writerows(rows_ep)

with open(os.path.join(OUT, 'cluster_counts.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['tag', 'images', 'thr', 'clusters', 'singleton_clusters', 'max_cluster'])
    w.writerows(all_rows)

print('\nALL DONE in %.1fs' % (time.time() - t_start), flush=True)
