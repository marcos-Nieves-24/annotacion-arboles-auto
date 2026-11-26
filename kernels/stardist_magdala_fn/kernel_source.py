# %% [markdown]
# # StarDist — copas perdidas (FN) en Magdala dev/holdout
# Diagnostico de falsos negativos: que copa del oro no fue detectada, donde
# esta (centro pixel, bbox, area), mejor IoU con alguna prediccion y si toca
# borde de tile. Genera overlays con FN en rojo. Sin CVAT writes.

# %%
import subprocess, sys
for pkg in ['stardist', 'csbdeep']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)

import glob, json, math, os, shutil
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
prob = float(getattr(model.thresholds, 'prob', 0.59))

# %%
def make_label(name):
    """Devuelve label image, lista de (ann_id, mask, poly_points) en orden de etiqueta."""
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
                        'points': pts.tolist(), 'area': int(m.sum())})
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
    up, ug, pairs_used = set(), set(), []
    for iou, i, j in pairs:
        if iou < 0.5 or i in up or j in ug:
            continue
        up.add(i); ug.add(j); pairs_used.append((iou, i, j))
    return pairs_used, pairs

def centroid(mask):
    yy, xx = np.nonzero(mask)
    return (round(float(xx.mean()), 1), round(float(yy.mean()), 1)) if len(xx) else (None, None)

# %%
report = {'model': 'limon-stardist-v1', 'prob_thresh': prob,
          'farm': 'Magdala', 'splits': ['dev', 'holdout'],
          'matching_iou': 0.5, 'images': len(names), 'nms': {}, 'items': []}

for nms in (0.02, 0.10):
    n_fn = n_fp = n_tp = 0
    for name in names:
        path = os.path.join(IMG_DIR, name)
        if not os.path.exists(path):
            continue
        raw = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
        x = normalize(raw, 1, 99.8, axis=(0, 1))
        lab, entries = make_label(name)
        golds = [e['mask'] for e in entries]
        out = model.predict_instances(x, prob_thresh=prob, nms_thresh=nms)
        pred_lab = out[0] if out[0].ndim == 2 else out[1]
        preds = masks_from(pred_lab)
        used, allpairs = match(preds, golds)
        matched_g = {j for _, _, j in used}
        matched_p = {i for _, i, _ in used}
        n_tp += len(used); n_fp += len(preds) - len(used); n_fn += len(golds) - len(used)

        # mejor IoU de cada copa del oro contra CUALQUIER prediccion
        best = {}
        for iou, i, j in allpairs:
            if j not in best or iou > best[j][0]:
                best[j] = (iou, i)

        vis = (np.clip(x * 255, 0, 255)).astype(np.uint8).copy()
        for e in entries:
            pts = np.asarray(e['points'], np.int32)
            cv2.polylines(vis, [pts], True, (60, 200, 90), 1)
        for i, m in enumerate(preds):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(vis, cnts, -1, (70, 150, 255) if i not in matched_p else (240, 220, 60), 1)

        for j, e in enumerate(entries):
            if j in matched_g:
                continue
            cx, cy = centroid(e['mask'])
            biou, bidx = best.get(j, (0.0, None))
            ys, xs = np.nonzero(e['mask'])
            kind = 'sub_threshold' if biou >= 0.1 else 'not_detected'
            item = {
                'nms': nms, 'image': name,
                'split': 'holdout' if name in hold_names else 'dev',
                'ann_id': e['ann_id'], 'centroid': [cx, cy],
                'bbox': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                'area_px': e['area'],
                'best_iou': round(float(biou), 4),
                'best_pred': bidx,
                'nearest_pred_dist': None,
                'edge_touch': bool(xs.min() <= 8 or ys.min() <= 8 or xs.max() >= 503 or ys.max() >= 503),
                'kind': kind,
            }
            if bidx is not None:
                pcx, pcy = centroid(preds[bidx])
                if pcx is not None:
                    item['nearest_pred_dist'] = round(math.hypot(cx - pcx, cy - pcy), 1)
            report['items'].append(item)
            cv2.circle(vis, (int(cx), int(cy)), 7, (255, 40, 40), -1)
            cv2.putText(vis, str(e['ann_id']), (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            pts = np.asarray(e['points'], np.int32)
            cv2.polylines(vis, [pts], True, (255, 40, 40), 2)

        tag = f"nms{nms}_{name.replace('.png','')}"
        Image.fromarray(vis).save(os.path.join(OUT, f'fn_{tag}.png'))

    report['nms'][str(nms)] = {'tp': n_tp, 'fp': n_fp, 'fn': n_fn}
    print('nms', nms, 'tp', n_tp, 'fp', n_fp, 'fn', n_fn, flush=True)

with open(os.path.join(OUT, 'magdala_fn_report.json'), 'w') as fh:
    json.dump(report, fh, indent=1)
print('items', len(report['items']), flush=True)
