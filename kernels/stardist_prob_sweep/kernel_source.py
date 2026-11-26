# %% [markdown]
# # StarDist — barrido de prob_thresh en Magdala dev/holdout
# Evalúa los mismos 26 frames usados en el análisis de FN, con el umbral de
# probabilidad variable. Objetivo: saber si las copas perdidas se deben a que
# el modelo no dispara (probabilidad baja) o a otra causa. Sin CVAT writes.

# %%
import subprocess, sys
for pkg in ['stardist', 'csbdeep']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)

import glob, json, math, os, shutil
import cv2, numpy as np
from PIL import Image
from csbdeep.utils import normalize
from stardist.models import StarDist2D

OUT = '/kaggle/working'
roots = glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/datasets/*/*')
ds = [p for p in roots
      if os.path.exists(os.path.join(p, 'coco.json'))
      and os.path.isdir(os.path.join(p, 'all', 'images'))]
assert len(ds) == 1, f'dataset ambiguo: {ds}'
BASE = ds[0]
DATA = os.path.join(BASE, 'all')
IMG_DIR = os.path.join(DATA, 'images')

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
try:
    quarantine = jload('quarantine.json')
    qids = {q['shape_id'] for q in quarantine} if isinstance(quarantine, list) else set()
except FileNotFoundError:
    qids = set()

names = sorted(n for n in set(by_id.values())
               if n.startswith('Limon_Magdala_') and (n in dev_names or n in hold_names))
print('imagenes Magdala dev+holdout', len(names), flush=True)

# %%
cfg = glob.glob('/kaggle/input/**/config.json', recursive=True)
wf = glob.glob('/kaggle/input/**/weights_best.h5', recursive=True)
assert cfg and wf
wroot = os.path.dirname(wf[0])
md = os.path.join(OUT, 'model', 'limon_stardist_v1')
os.makedirs(md, exist_ok=True)
for f in ('config.json', 'weights_best.h5', 'thresholds.json'):
    p = os.path.join(wroot, f)
    if os.path.exists(p):
        shutil.copyfile(p, os.path.join(md, f))
model = StarDist2D(None, name='limon_stardist_v1', basedir=os.path.join(OUT, 'model'))
base_prob = float(getattr(model.thresholds, 'prob', 0.59))
print('prob baseline', base_prob, flush=True)

# %%
def make_label(name):
    """label image +lista de (ann_id, mask, seg_flat) para clasificar FN por borde."""
    lab = np.zeros((512, 512), dtype=np.int32)
    gid = name_to_id[name]
    anns = [a for a in anns_by_id.get(gid, [])
            if not a.get('iscrowd') and a.get('id') not in qids]
    anns.sort(key=lambda a: -float(a.get('area', 0)))
    entries = []
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
        entries.append({'ann_id': a.get('id'), 'mask': m.astype(bool),
                        'flat': [float(v) for v in ring]})
    return lab, entries

def masks_from(lab):
    return [lab == v for v in np.unique(lab) if v > 0]

def match(preds, golds):
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            u = np.count_nonzero(p | g)
            if u:
                pairs.append((np.count_nonzero(p & g) / u, i, j))
    pairs.sort(reverse=True)
    up, ug, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < 0.5 or i in up or j in ug:
            continue
        up.add(i); ug.add(j); tp += 1
    return tp, len(preds) - tp, len(golds) - tp, ug

def border_vertices(flat, tol=2.0):
    return sum(1 for k in range(0, len(flat), 2)
               if flat[k] <= tol or flat[k+1] <= tol
               or flat[k] >= 512 - tol or flat[k+1] >= 512 - tol)

# %%
X = []
for name in names:
    p = os.path.join(IMG_DIR, name)
    raw = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
    X.append((name, normalize(raw, 1, 99.8, axis=(0, 1))))
print('imagenes cargadas', len(X), flush=True)

PROBS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, round(base_prob, 4)]
NMS = [0.02, 0.10]
report = {'model': 'limon-stardist-v1', 'farm': 'Magdala',
          'split': 'dev+holdout', 'n_images': len(names),
          'base_prob': base_prob, 'probs': PROBS, 'nms': NMS,
          'matching_iou': 0.5, 'results': {}, 'fn_details': {}}

for nms in NMS:
    for prob in PROBS:
        tp = fp = fn = 0
        fn_border = {'0': 0, '1': 0, '2': 0, '>=3': 0}
        fn_ids = []
        for name, x in X:
            _, entries = make_label(name)
            golds = [e['mask'] for e in entries]
            out = model.predict_instances(x, prob_thresh=prob, nms_thresh=nms)
            lab = out[0] if out[0].ndim == 2 else out[1]
            preds = masks_from(lab)
            a, b, c, matched_g = match(preds, golds)
            tp += a; fp += b; fn += c
            for j, e in enumerate(entries):
                if j not in matched_g:
                    nv = border_vertices(e['flat'])
                    key = '>=3' if nv >= 3 else str(nv)
                    fn_border[key] += 1
                    fn_ids.append({'image': name, 'ann_id': e['ann_id'],
                                   'border_vertices': nv})
        p_ = tp / max(tp + fp, 1)
        r_ = tp / max(tp + fn, 1)
        f1 = 2 * p_ * r_ / max(p_ + r_, 1e-9)
        key = f'nms{nms}_prob{prob}'
        report['results'][key] = {'prob_thresh': prob, 'nms': nms,
                                  'tp': tp, 'fp': fp, 'fn': fn,
                                  'precision': round(p_, 4), 'recall': round(r_, 4),
                                  'F1': round(f1, 4),
                                  'fn_border': fn_border}
        report['fn_details'][key] = fn_ids
        print(key, 'TP', tp, 'FP', fp, 'FN', fn,
              'P', round(p_, 3), 'R', round(r_, 3), 'F1', round(f1, 3),
              'borde', fn_border, flush=True)

json.dump(report, open(os.path.join(OUT, 'stardist_prob_sweep.json'), 'w'), indent=1)
print('LISTO', len(report['results']), 'configuraciones', flush=True)
