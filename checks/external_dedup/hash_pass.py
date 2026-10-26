# -*- coding: utf-8 -*-
"""Pass 1: compute rotation+mirror-invariant 256-bit dHash for external + project images.

Read-only w.r.t. the datasets. Writes a cache under checks/external_dedup/.
"""
import os, sys, time
import numpy as np
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath('.')
EXT_IMG = os.path.join(ROOT, 'data', 'external', 'rice-leaf-diseases', 'rice', 'images')
EXT_LBL = os.path.join(ROOT, 'data', 'external', 'rice-leaf-diseases', 'rice', 'labels')
PRJ_IMG = os.path.join(ROOT, 'datasets', 'dataset_fieldlike', 'images')
OUT = os.path.join(ROOT, 'checks', 'external_dedup')
os.makedirs(OUT, exist_ok=True)

EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def collect(img_root):
    items = []
    for split in ('train', 'val'):
        d = os.path.join(img_root, split)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(EXTS):
                continue
            stem = os.path.splitext(name)[0]
            items.append((split, name, stem, os.path.join(d, name)))
    return items

def read_classes(root_lbl, split, stem):
    p = os.path.join(root_lbl, split, stem + '.txt')
    if not os.path.isfile(p):
        return (-1,)
    ids = set()
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(int(float(line.split()[0])))
            except Exception:
                pass
    return tuple(sorted(ids)) if ids else (-1,)

def dhash256(img):
    """8 orientation variants of a 256-bit difference hash (16x16)."""
    out = np.empty((8, 4), dtype=np.uint64)
    k = 0
    for mirror in (False, True):
        base = ImageOps.mirror(img) if mirror else img
        for rot in (0, 1, 2, 3):
            im = base if rot == 0 else base.rotate(90 * rot, expand=True)
            g = im.convert('L').resize((17, 16), Image.BILINEAR)
            a = np.asarray(g, dtype=np.int16)
            bits = (a[:, 1:] > a[:, :-1]).reshape(-1)          # 256 bits
            packed = np.packbits(bits)                          # 32 bytes
            out[k] = packed.view(np.uint64)                     # 4 x uint64
            k += 1
    return out

def work(item):
    split, name, stem, path = item
    try:
        with Image.open(path) as im:
            im.load()
            h = dhash256(im)
        return h
    except Exception as e:
        sys.stderr.write('FAIL %s: %s\n' % (path, e))
        return np.zeros((8, 4), dtype=np.uint64)

def hash_set(items, tag, workers=8):
    t0 = time.time()
    hashes = np.zeros((len(items), 8, 4), dtype=np.uint64)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, h in enumerate(ex.map(work, items, chunksize=32)):
            hashes[i] = h
            done += 1
            if done % 1000 == 0:
                print('  %s: %d/%d  (%.1fs)' % (tag, done, len(items), time.time() - t0), flush=True)
    print('  %s: done %d in %.1fs' % (tag, done, time.time() - t0), flush=True)
    return hashes

# ---- external ----
ext_items = collect(EXT_IMG)
ext_cls = [read_classes(EXT_LBL, s, st) for (s, n, st, p) in ext_items]
print('external images: %d' % len(ext_items), flush=True)
ext_h = hash_set(ext_items, 'external')

np.savez_compressed(
    os.path.join(OUT, 'ext_hash.npz'),
    hashes=ext_h,
    split=np.array([x[0] for x in ext_items]),
    name=np.array([x[1] for x in ext_items]),
    stem=np.array([x[2] for x in ext_items]),
    cls=np.array([','.join(str(c) for c in x) for x in ext_cls]),
)

# ---- project (dataset_fieldlike) ----
prj_items = collect(PRJ_IMG)
print('project images: %d' % len(prj_items), flush=True)
prj_h = hash_set(prj_items, 'project')

np.savez_compressed(
    os.path.join(OUT, 'prj_hash.npz'),
    hashes=prj_h,
    split=np.array([x[0] for x in prj_items]),
    name=np.array([x[1] for x in prj_items]),
    stem=np.array([x[2] for x in prj_items]),
)
print('OK', flush=True)
