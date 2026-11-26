# %% [markdown]
# # StarDist v1 — matriz de confusión contra gold_v2 por finca
# Evalúa los pesos limon-stardist-v1 sobre las 405 imagenes del oro.
# Matrices a nivel de pixel (2x2) y desglose de instancias (TP/FP/FN, IoU>=0.5),
# separando por finca (LasVillas / Magdala) y por split (train / dev / holdout).
# Sin reentrenamiento, sin SAM, sin centroides, sin CVAT writes.

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
audited_names = {
    'Limon_LasVillas_00509.png', 'Limon_LasVillas_00196.png',
    'Limon_LasVillas_00499.png', 'Limon_LasVillas_00563.png',
    'Limon_LasVillas_00519.png', 'Limon_LasVillas_00408.png',
    'Limon_LasVillas_00626.png', 'Limon_Magdala_00866.png',
    'Limon_Magdala_00946.png',
}
all_names = sorted(set(by_id.values()))
train_names = sorted(set(all_names) - dev_names - hold_names - audited_names)

try:
    quarantine = jload('quarantine.json')
    qids = {q['shape_id'] for q in quarantine} if isinstance(quarantine, list) else set()
except FileNotFoundError:
    qids = set()

def split_of(name):
    if name in hold_names:
        return 'holdout'
    if name in dev_names:
        return 'dev'
    if name in audited_names:
        return 'audited_excluded'
    return 'train'

def farm_of(name):
    parts = name.split('_')
    return parts[1] if len(parts) > 1 else 'unknown'

# %%
# pesos: mismo modelo exacto del pipeline StarDist
cfg = glob.glob('/kaggle/input/**/config.json', recursive=True)
wf = glob.glob('/kaggle/input/**/weights_best.h5', recursive=True)
assert cfg and wf, f'pesos no encontrados cfg={cfg} w={wf}'
wroot = os.path.dirname(wf[0])
md = os.path.join(OUT, 'model', 'limon_stardist_v1')
os.makedirs(md, exist_ok=True)
for f in ('config.json', 'weights_best.h5', 'thresholds.json'):
    p = os.path.join(wroot, f)
    if os.path.exists(p):
        shutil.copyfile(p, os.path.join(md, f))
model = StarDist2D(None, name='limon_stardist_v1', basedir=os.path.join(OUT, 'model'))
prob = float(getattr(model.thresholds, 'prob', 0.59))
print('prob', prob, 'imagenes', len(all_names), flush=True)

# %%
def make_label_image(name):
    lab = np.zeros((512, 512), dtype=np.int32)
    gid = name_to_id[name]
    anns = [a for a in anns_by_id.get(gid, [])
            if not a.get('iscrowd') and a.get('id') not in qids]
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

def masks_from(lab):
    return [lab == v for v in np.unique(lab) if v > 0]

def match_masks(preds, golds, thr=0.5):
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            u = np.count_nonzero(p | g)
            if u:
                pairs.append((np.count_nonzero(p & g) / u, i, j))
    pairs.sort(reverse=True)
    up, ug, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < thr or i in up or j in ug:
            continue
        up.add(i); ug.add(j); tp += 1
    return tp, len(preds) - tp, len(golds) - tp

# %%
def new_bucket():
    return {'n_img': 0, 'tp': 0, 'fp': 0, 'fn': 0,
            'tp_px': 0, 'fp_px': 0, 'fn_px': 0, 'tn_px': 0}

def add(b, tp, fp, fn, tp_px, fp_px, fn_px, tn_px):
    b['n_img'] += 1
    b['tp'] += tp; b['fp'] += fp; b['fn'] += fn
    b['tp_px'] += tp_px; b['fp_px'] += fp_px
    b['fn_px'] += fn_px; b['tn_px'] += tn_px

def summarize(b):
    tp, fp, fn = b['tp'], b['fp'], b['fn']
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    tpx, fpx, fnpx = b['tp_px'], b['fp_px'], b['fn_px']
    iou_px = tpx / max(tpx + fpx + fnpx, 1)
    return dict(b, precision=round(p, 4), recall=round(r, 4), F1=round(f1, 4),
                iou_px=round(iou_px, 4))

report = {
    'model': 'marcosalvarez234/limon-stardist-v1-weights',
    'dataset': BASE,
    'prob_thresh': prob,
    'iou_match_threshold': 0.5,
    'n_images': len(all_names),
    'splits': {'train': len(train_names), 'dev': len(dev_names),
               'holdout': len(hold_names), 'audited_excluded': len(audited_names)},
    'farms': sorted({farm_of(n) for n in all_names}),
    'buckets': {},
}
per_image = []

for nms in (0.02, 0.10):
    buckets = {'ALL': new_bucket()}
    for sp in ('train', 'dev', 'holdout', 'audited_excluded'):
        buckets[sp] = new_bucket()
    for farm in report['farms']:
        buckets[f'farm:{farm}'] = new_bucket()
        for sp in ('train', 'dev', 'holdout'):
            buckets[f'{farm}/{sp}'] = new_bucket()

    for k, name in enumerate(all_names):
        p = os.path.join(IMG_DIR, name)
        if not os.path.exists(p):
            continue
        raw = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
        x = normalize(raw, 1, 99.8, axis=(0, 1))
        lab_gt = make_label_image(name)
        golds = masks_from(lab_gt)
        out = model.predict_instances(x, prob_thresh=prob, nms_thresh=nms)
        pred_lab = out[0] if out[0].ndim == 2 else out[1]
        preds = masks_from(pred_lab)
        tp, fp, fn = match_masks(preds, golds)
        gu = np.zeros((512, 512), bool)
        pu = np.zeros((512, 512), bool)
        for m in golds:
            gu |= m
        for m in preds:
            pu |= m
        tp_px = int(np.count_nonzero(gu & pu))
        fp_px = int(np.count_nonzero(~gu & pu))
        fn_px = int(np.count_nonzero(gu & ~pu))
        tn_px = int(512 * 512 - tp_px - fp_px - fn_px)
        sp, farm = split_of(name), farm_of(name)
        for key in ('ALL', sp, f'farm:{farm}', f'{farm}/{sp}'):
            if key in buckets:
                add(buckets[key], tp, fp, fn, tp_px, fp_px, fn_px, tn_px)
        per_image.append({'nms': nms, 'image': name, 'split': sp, 'farm': farm,
                          'tp': tp, 'fp': fp, 'fn': fn,
                          'tp_px': tp_px, 'fp_px': fp_px,
                          'fn_px': fn_px, 'tn_px': tn_px})
        if (k + 1) % 50 == 0:
            print('nms', nms, 'procesadas', k + 1, '/', len(all_names), flush=True)

    report['buckets'][f'nms_{nms}'] = {k: summarize(v) for k, v in buckets.items()}

report['per_image'] = per_image
with open(os.path.join(OUT, 'stardist_confusion.json'), 'w') as fh:
    json.dump(report, fh, indent=1)
print('LISTO buckets', len(report['buckets']), 'filas', len(per_image), flush=True)
