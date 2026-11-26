# %% [markdown]
# # StarDist v1 — copas star-convex sobre el oro (Kaggle T4)
#
# Entrena StarDist2D desde cero con las máscaras del oro. NMS bajo para que
# las copas no se solapen. Sin SAM, sin centroides, sin CVAT writes.

# %%
import subprocess
import sys
for pkg in ['stardist', 'csbdeep']:
    try:
        __import__(pkg)
        print(pkg, 'presente')
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)
        print(pkg, 'instalado')

import glob
import json
import math
import os
import random

import cv2
import numpy as np
from PIL import Image

from csbdeep.utils import normalize
from stardist import fill_label_holes, gputools_available
from stardist.models import Config2D, StarDist2D

import torch

print('cuda:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'se requiere GPU T4'

roots = glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/datasets/*/*')
ds = [p for p in roots
      if os.path.exists(os.path.join(p, 'coco.json'))
      and os.path.isdir(os.path.join(p, 'all', 'images'))]
assert len(ds) == 1, f'dataset ambiguo: {ds}'
BASE = ds[0]
DATA = os.path.join(BASE, 'all')
IMG_DIR = os.path.join(DATA, 'images')
OUT = '/kaggle/working'
report = {'dataset': BASE, 'pipeline': 'StarDist2D from scratch on gold_v2'}
print('BASE =', BASE)

# %%
def jload(fn):
    for d in (DATA, BASE):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
    raise FileNotFoundError(fn)

coco = jload('coco.json')
by_id = {im['id']: im['file_name'] for im in coco['images']}
name_to_id = {v: k for k, v in by_id.items()}
anns_by_id = {}
for a in coco['annotations']:
    anns_by_id.setdefault(a['image_id'], []).append(a)

dev = jload('dev.json')
dev_ids = set(dev.get('A', []) + dev.get('B', []))
dev_names = {by_id[i] for i in dev_ids if i in by_id}
hold_raw = jload('holdout.json')
hold_names = set(hold_raw['holdout'] if isinstance(hold_raw, dict) else hold_raw)
audited_names = {
    'Limon_LasVillas_00509.png', 'Limon_LasVillas_00196.png',
    'Limon_LasVillas_00499.png', 'Limon_LasVillas_00563.png',
    'Limon_LasVillas_00519.png', 'Limon_LasVillas_00408.png',
    'Limon_LasVillas_00626.png', 'Limon_Magdala_00866.png',
    'Limon_Magdala_00946.png',
}
all_names = set(by_id.values())
train_names = sorted(all_names - dev_names - hold_names - audited_names)
report['splits'] = {'train': len(train_names), 'dev': len(dev_names),
                    'holdout': len(hold_names), 'audited_excluded': len(audited_names)}
print(report['splits'])

# %%
# quarantine lives in the dataset root or BASE; exclude those shape ids.
try:
    quarantine = jload('quarantine.json')
    qids = {q['shape_id'] for q in quarantine} if isinstance(quarantine, list) else set()
except FileNotFoundError:
    qids = set()
print('quarantine ids', len(qids))


def make_label_image(name):
    """Build a StarDist label image (YX int32) from COCO polygons."""
    gid = name_to_id[name]
    lab = np.zeros((512, 512), dtype=np.int32)
    anns = anns_by_id.get(gid, [])
    # skip iscrowd (tile-edge clipped) and quarantine splinters
    anns = [a for a in anns if not a.get('iscrowd') and a.get('id') not in qids]
    # larger crowns first, so a smaller one overwrites only its own pixels
    anns.sort(key=lambda a: -float(a.get('area', 0)))
    n = 0
    for a in anns:
        seg = a.get('segmentation', [])
        if not seg:
            continue
        ring = seg[0] if (isinstance(seg[0], list) and seg) else seg
        pts = np.asarray(ring, dtype=np.int32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        m = np.zeros((512, 512), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        if m.sum() < 200:
            continue
        n += 1
        lab[m > 0] = n
    return lab


def load_xy(names):
    X, Y = [], []
    skipped = 0
    for name in sorted(names):
        p = os.path.join(IMG_DIR, name)
        if not os.path.exists(p):
            skipped += 1
            continue
        img = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
        lab = make_label_image(name)
        if lab.max() == 0:
            skipped += 1
            continue
        img = normalize(img, 1, 99.8, axis=(0, 1))
        lab = fill_label_holes(lab)
        X.append(img)
        Y.append(lab)
    print('loaded', len(X), 'skipped', skipped)
    return X, Y

X_train, Y_train = load_xy(train_names)
X_dev, Y_dev = load_xy(dev_names)
X_hold, Y_hold = load_xy(hold_names)
report['n_train'] = len(X_train)
report['n_dev'] = len(X_dev)
report['n_holdout'] = len(X_hold)

# star-convex reconstruction sanity check with 32 rays on a sample
def star_iou(lab, n_rays=32):
    idx = np.argwhere(lab > 0)
    if len(idx) == 0:
        return None
    i = np.random.randint(len(idx))
    y, x = idx[i]
    val = lab[y, x]
    m = (lab == val)
    cx = int(np.argwhere(m)[:, 1].mean())
    cy = int(np.argwhere(m)[:, 0].mean())
    rays = []
    for t in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        r = 0
        for d in range(1, 130):
            xx, yy = int(cx + d * math.cos(t)), int(cy + d * math.sin(t))
            if not (0 <= xx < 512 and 0 <= yy < 512) or not m[yy, xx]:
                break
            r = d
        rays.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    sp = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(sp, [np.asarray(rays, np.int32)], 1)
    u = np.count_nonzero(sp | m)
    return np.count_nonzero(sp & m) / u if u else None

np.random.seed(7)
sious = [v for Y in (Y_train[:40], Y_dev[:10]) for lab in Y
         if (v := star_iou(lab)) is not None]
report['star_iou_32rays_median'] = round(float(np.median(sious)), 4) if sious else None
print('star iou 32 rays', report['star_iou_32rays_median'], 'n', len(sious))

# %%
conf = Config2D(
    n_rays=32,
    grid=(2, 2),
    n_channel_in=3,
    backbone='unet',
    unet_n_depth=3,
    unet_n_filter_base=32,
    train_patch_size=(256, 256),
    train_batch_size=8,
    train_epochs=120,
    train_steps_per_epoch=100,
    train_learning_rate=3e-4,
    train_foreground_only=0.9,
    train_dist_loss='mae',
    train_loss_weights=[1, 0.2],
    train_reduce_lr={'factor': 0.5, 'patience': 30},
    use_gpu=bool(gputools_available()),
)
report['config'] = {
    'n_rays': 32, 'grid': [2, 2], 'n_channel_in': 3,
    'patch': [256, 256], 'batch': 8, 'epochs': 120,
    'steps': 100, 'lr': 3e-4, 'foreground_only': 0.9,
    'dist_loss': 'mae', 'loss_weights': [1, 0.2],
    'use_gpu': bool(gputools_available()),
}
model = StarDist2D(conf, name='limon_stardist_v1', basedir=OUT)
model.train(X_train, Y_train, validation_data=(X_dev, Y_dev),
            augmenter=None, epochs=120, steps_per_epoch=100)
print('training done')

# %%
prob_thresh, nms_opt = model.optimize_thresholds(X_dev, Y_dev)
report['thresholds'] = {'prob_optimized': float(prob_thresh),
                        'nms_optimized': float(nms_opt)}
print('thresholds', report['thresholds'])


def predict_mask(img, prob, nms):
    out = model.predict_instances(img, prob_thresh=prob, nms_thresh=nms)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0] if out[0].ndim == 2 else out[1]
    return out


def match_masks(preds, golds, thr=0.5):
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            inter = np.count_nonzero(p & g)
            union = np.count_nonzero(p | g)
            if union:
                pairs.append((inter / union, i, j))
    pairs.sort(reverse=True)
    up, ug, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < thr or i in up or j in ug:
            continue
        up.add(i)
        ug.add(j)
        tp += 1
    return tp, len(preds) - tp, len(golds) - tp


def masks_from_label(lab):
    out = []
    for v in np.unique(lab):
        if v > 0:
            out.append(lab == v)
    return out


def evaluate(X, Y, prob, nms, max_overlay=6, split='dev'):
    tp = fp = fn = 0
    iou_pairs = 0
    n_gt = n_pred = 0
    area_ratios = []
    circs = []
    shown = 0
    for img, lab in zip(X, Y):
        pred_lab = predict_mask(img, prob, nms)
        preds = masks_from_label(pred_lab)
        golds = masks_from_label(lab)
        a, b, c = match_masks(preds, golds)
        tp += a
        fp += b
        fn += c
        n_gt += len(golds)
        n_pred += len(preds)
        # pairwise overlap among predictions
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                u = np.count_nonzero(preds[i] | preds[j])
                if u and np.count_nonzero(preds[i] & preds[j]) / u > 0.05:
                    iou_pairs += 1
        for m in preds:
            area = float(m.sum())
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cc = max(cnts, key=cv2.contourArea)
                per = cv2.arcLength(cc, True)
                if per:
                    circs.append(4 * math.pi * area / (per * per))
        if shown < max_overlay:
            vis = (np.clip(img * 255, 0, 255)).astype(np.uint8).copy()
            for m in preds:
                vis[m] = (0.55 * vis[m] + 0.45 * np.array([70, 220, 110])).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(OUT, f'overlay_{split}_{shown}.png'))
            shown += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {'precision': round(float(p), 4), 'recall': round(float(r), 4),
            'F1': round(float(f1), 4), 'tp': tp, 'fp': fp, 'fn': fn,
            'n_pred': n_pred, 'n_gt': n_gt,
            'pares_iou_gt_005': iou_pairs,
            'circ_median': round(float(np.median(circs)), 3) if circs else None}

# %%
report['dev'] = {}
report['holdout'] = {}
for nms in (0.02, 0.05, 0.10):
    m = evaluate(X_dev, Y_dev, prob_thresh, nms, split='dev')
    report['dev'][f'nms_{nms}'] = m
    print('dev nms', nms, m, flush=True)

best_nms = min((0.02, 0.05, 0.10),
               key=lambda n: (-report['dev'][f'nms_{n}']['F1'],
                              report['dev'][f'nms_{n}']['pares_iou_gt_005']))
report['best_nms'] = best_nms
report['holdout'] = evaluate(X_hold, Y_hold, prob_thresh, best_nms, split='holdout')
print('holdout', report['holdout'], flush=True)

report['gates'] = {
    'E2_dev_F1_ge_0.931': report['dev'][f'nms_{best_nms}']['F1'] >= 0.931,
    'E2_holdout_F1_ge_0.898': report['holdout']['F1'] >= 0.898,
    'overlap_zero_dev': report['dev'][f'nms_{best_nms}']['pares_iou_gt_005'] == 0,
    'overlap_zero_holdout': report['holdout']['pares_iou_gt_005'] == 0,
}
with open(os.path.join(OUT, 'stardist_report.json'), 'w') as fh:
    json.dump(report, fh, indent=1)
print(json.dumps(report, indent=1))
