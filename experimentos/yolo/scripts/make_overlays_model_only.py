"""Overlays with ONLY the T1b model output (no CVAT gold, no auto polygons).

conf=0.25, imgsz=640, NMS 0.35. Red=box, green=mask, label=confidence.
Output: data/fase4/t1b_overlays_model_only/overlays/ + contact sheets + report.json
"""

import json, os, glob
import numpy as np, cv2
from ultralytics import YOLO

ROOT = '/home/biohacker24/annotacion-arboles-auto'
OUT = f'{ROOT}/data/fase4/t1b_overlays_model_only'
os.makedirs(f'{OUT}/overlays', exist_ok=True)
IMG = glob.glob('/tmp/user/1000/opencode/vc/**/images', recursive=True)[0]
W = '/tmp/user/1000/opencode/wm/model_final.pt'
CONF = 0.25

C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
H = json.load(open(f'{ROOT}/data/gold_v2/holdout.json'))
hold = set(H['holdout'] if isinstance(H, dict) else [h['file_name'] for h in H])
byid = {i['id']: i['file_name'] for i in C['images']}

frames = []
for iid in D['A'] + D['B']:
    fn = byid.get(iid)
    if fn and fn not in hold:
        frames.append(('dev', fn))
for fn in sorted(hold):
    if not any(f == fn for _, f in frames):
        frames.append(('holdout', fn))

model = YOLO(W)
report = []
for split, fn in frames:
    p = os.path.join(IMG, fn)
    if not os.path.exists(p):
        continue
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    r = model.predict(img, imgsz=640, conf=CONF, iou=0.35, verbose=False)[0]
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    nb = len(r.boxes) if r.boxes is not None else 0
    confs = r.boxes.conf.tolist() if nb else []
    for k in range(nb):
        x0, y0, x1, y1 = [int(v) for v in r.boxes.xyxy[k].tolist()]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(vis, f'{confs[k]:.2f}', (x0, max(12, y0 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
    if r.masks is not None and len(r.masks):
        md = r.masks.data.float().cpu().numpy()
        for k in range(md.shape[0]):
            mm = md[k] > 0.5
            if mm.shape != (512, 512):
                mm = cv2.resize(mm.astype('uint8'), (512, 512)) > 0.5
            cnts, _ = cv2.findContours(mm.astype('uint8'), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(f'{OUT}/overlays/{split}_{fn}', vis)
    report.append({'split': split, 'file': fn, 'n_det': nb})

json.dump(report, open(f'{OUT}/report.json', 'w'), indent=1)
tw = th = 256
items = report[:30]
rows = (len(items) + 4) // 5
sheet = np.full((rows * th, 5 * tw, 3), 40, np.uint8)
for k, r in enumerate(items):
    im = cv2.imread(f"{OUT}/overlays/{r['split']}_{r['file']}")
    if im is None:
        continue
    im = cv2.resize(im, (tw, th))
    cv2.putText(im, f"{r['file'][-9:-4]} n{r['n_det']}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    sheet[(k // 5) * th:(k // 5 + 1) * th, (k % 5) * tw:(k % 5 + 1) * tw] = im
cv2.imwrite(f'{OUT}/contact_model_only.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f'DONE: {len(report)} overlays (model only), conf={CONF}')
