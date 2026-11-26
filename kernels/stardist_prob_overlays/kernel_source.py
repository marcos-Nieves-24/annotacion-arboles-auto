# %% [markdown]
# # StarDist — overlays por prob_thresh (Magdala dev/holdout)
# 6 imagenes con mas FN en el baseline, 7 umbrales, NMS 0.10.
# Rojo = copa del oro no detectada; amarillo = TP; azul = prediccion no match;
# verde = contorno del oro. Sin CVAT writes.

# %%
import subprocess, sys
for pkg in ['stardist', 'csbdeep']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)

import glob, json, os, shutil
import cv2, numpy as np
from PIL import Image, ImageDraw
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

NAMES = ['Limon_Magdala_00946.png', 'Limon_Magdala_01027.png',
         'Limon_Magdala_00866.png', 'Limon_Magdala_01012.png',
         'Limon_Magdala_01042.png', 'Limon_Magdala_01043.png']

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
PROBS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, round(base_prob, 4)]
print('probs', PROBS, flush=True)

# %%
def make_label(name):
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
                        'points': pts})
    return entries

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
    up, ug = set(), set()
    for iou, i, j in pairs:
        if iou < 0.5 or i in up or j in ug:
            continue
        up.add(i); ug.add(j)
    return up, ug

summary = {}
for prob in PROBS:
    tag = f'{prob:.4f}'
    summary[tag] = {}
    for name in NAMES:
        raw = np.asarray(Image.open(os.path.join(IMG_DIR, name)).convert('RGB'), dtype=np.uint8)
        x = normalize(raw, 1, 99.8, axis=(0, 1))
        entries = make_label(name)
        golds = [e['mask'] for e in entries]
        out = model.predict_instances(x, prob_thresh=prob, nms_thresh=0.10)
        lab = out[0] if out[0].ndim == 2 else out[1]
        preds = masks_from(lab)
        mp, mg = match(preds, golds)

        vis = (np.clip(x * 255, 0, 255)).astype(np.uint8).copy()
        for e in entries:
            cv2.polylines(vis, [e['points']], True, (60, 200, 90), 1)
        for i, m in enumerate(preds):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(vis, cnts, -1,
                                 (70, 150, 255) if i not in mp else (240, 220, 60), 1)
        for j, e in enumerate(entries):
            if j not in mg:
                yy, xx = np.nonzero(e['mask'])
                if len(xx) == 0:
                    continue
                cx, cy = int(xx.mean()), int(yy.mean())
                cv2.polylines(vis, [e['points']], True, (255, 40, 40), 2)
                cv2.circle(vis, (cx, cy), 6, (255, 40, 40), -1)
        pil = Image.fromarray(vis)
        d = ImageDraw.Draw(pil)
        d.text((6, 6), f'prob={tag}  pred={len(preds)}  FN={len(golds)-len(mg)}  FP={len(preds)-len(mp)}',
               fill=(255, 255, 255))
        pil.save(os.path.join(OUT, f'ov_prob{tag}_{name}'))

        summary[tag][name] = {'n_pred': len(preds), 'n_gt': len(golds),
                              'fn': len(golds) - len(mg), 'fp': len(preds) - len(mp)}
    print('prob', tag, 'FN total', sum(v['fn'] for v in summary[tag].values()),
          'FP total', sum(v['fp'] for v in summary[tag].values()), flush=True)

json.dump(summary, open(os.path.join(OUT, 'prob_overlays_summary.json'), 'w'), indent=1)
print('LISTO', len(PROBS) * len(NAMES), 'overlays', flush=True)
