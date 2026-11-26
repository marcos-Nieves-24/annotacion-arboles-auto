"""Select hard training frames: gold crowns with NO matching YOLO detection.

Runs T1b (conf 0.25) over the 347 TRAIN frames, rasterizes gold non-iscrowd
masks, and flags frames where >=1 gold crown has no detection overlapping it.
Those frames will be over-sampled x3 in the next training round (frente A).

Read-only, local CPU. Output: data/fase4/hard_train.json
"""

import json, os, glob
import numpy as np, cv2
from ultralytics import YOLO

ROOT = '/home/biohacker24/annotacion-arboles-auto'
IMG_DIRS = [f'{ROOT}/data/r1/images/villas', f'{ROOT}/data/r1/images/magdala',
            f'{ROOT}/data/gold_v1/images']
W = '/tmp/user/1000/opencode/wm/model_final.pt'
OUT = f'{ROOT}/data/fase4/hard_train.json'

C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
H = json.load(open(f'{ROOT}/data/gold_v2/holdout.json'))
hold = set(H['holdout'] if isinstance(H, dict) else [h['file_name'] for h in H])
byid = {i['id']: i['file_name'] for i in C['images']}
dev = {byid[i] for i in D['A'] + D['B'] if i in byid}
train = [i for i in C['images'] if i['file_name'] not in hold and i['file_name'] not in dev]


def load_img(fn):
    for d in IMG_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return None


def gold_masks(iid):
    out = []
    for a in C['annotations']:
        if a['image_id'] == iid and not a['iscrowd'] and a.get('segmentation'):
            pts = np.array(a['segmentation'][0], float).reshape(-1, 2).astype(int)
            m = np.zeros((512, 512), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            out.append(m.astype(bool))
    return out


model = YOLO(W)
hard = []
rows = []
for info in train:
    fn = info['file_name']
    img = load_img(fn)
    if img is None:
        continue
    r = model.predict(img, imgsz=640, conf=0.25, iou=0.35, verbose=False)[0]
    preds = []
    if r.masks is not None and len(r.masks):
        md = r.masks.data.float().cpu().numpy()
        for k in range(md.shape[0]):
            m = md[k] > 0.5
            if m.shape != (512, 512):
                m = cv2.resize(m.astype('uint8'), (512, 512)) > 0.5
            preds.append(m)
    refs = gold_masks(info['id'])
    unmatched = 0
    for g in refs:
        best = 0
        for p in preds:
            it = np.logical_and(g, p).sum()
            un = np.logical_or(g, p).sum()
            if un:
                best = max(best, it / un)
        if best < 0.25:
            unmatched += 1
    rows.append({'file': fn, 'gold': len(refs), 'pred': len(preds), 'unmatched': unmatched})
    if unmatched > 0:
        hard.append({'file': fn, 'gold': len(refs), 'pred': len(preds),
                     'unmatched': unmatched, 'recall_frame': round(1 - unmatched/max(len(refs),1), 3)})

json.dump({'n_train': len(rows), 'n_hard': len(hard),
           'total_gold': sum(r['gold'] for r in rows),
           'total_unmatched': sum(r['unmatched'] for r in rows),
           'hard_frames': hard, 'all_frames': rows},
          open(OUT, 'w'), indent=1)

print(f'train frames procesados: {len(rows)}')
print(f'copas gold en train: {sum(r["gold"] for r in rows)}')
print(f'copas sin detectar: {sum(r["unmatched"] for r in rows)}')
print(f'frames con >=1 copa sin detectar (hard): {len(hard)}')
print(f'ejemplos: {[h["file"][-9:-4] for h in hard[:8]]}')
