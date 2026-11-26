"""Test YOLO T1b (conf 0.25, imgsz 640, NMS 0.35) on Villas frames 270-400.

These are UNLABELED frames (not in gold_v2): inference only, no accuracy metric.
Produces overlays + per-frame counts + a contact sheet. No CVAT writes.
"""

import json, os
import numpy as np, cv2
from ultralytics import YOLO

ROOT = '/home/biohacker24/annotacion-arboles-auto'
OUT = f'{ROOT}/data/fase4/test_villas_270_400'
os.makedirs(f'{OUT}/overlays', exist_ok=True)
IMG = f'{ROOT}/data/r1/images/villas'
W = '/tmp/user/1000/opencode/wm/model_final.pt'
CONF = 0.25

meta = json.load(open(f'{ROOT}/data/raw/villas/meta.json'))
names = [f['name'] for f in meta['frames']]
frames = [(i, names[i]) for i in range(270, 401) if i < len(names)]

model = YOLO(W)
report = []
for idx, fn in frames:
    p = os.path.join(IMG, fn)
    if not os.path.exists(p):
        continue
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    r = model.predict(img, imgsz=640, conf=CONF, iou=0.35, verbose=False)[0]
    nb = len(r.boxes) if r.boxes is not None else 0
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for k in range(nb):
        x0, y0, x1, y1 = [int(v) for v in r.boxes.xyxy[k].tolist()]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
    if r.masks is not None and len(r.masks):
        md = r.masks.data.float().cpu().numpy()
        for k in range(md.shape[0]):
            m = md[k] > 0.5
            if m.shape != (512, 512):
                m = cv2.resize(m.astype('uint8'), (512, 512)) > 0.5
            cnts, _ = cv2.findContours(m.astype('uint8'), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 1)
    cv2.imwrite(f'{OUT}/overlays/{fn}', vis)
    report.append({'frame_idx': idx, 'file': fn, 'n_det': nb})

json.dump({'config': {'model': 'yolo11n-seg T1b', 'conf': CONF, 'imgsz': 640, 'iou': 0.35},
           'frames': report}, open(f'{OUT}/report.json', 'w'), indent=1)

# contact sheet
items = report[:30]
tw = th = 220
rows = (len(items) + 5) // 6
sheet = np.full((rows * th, 6 * tw, 3), 40, np.uint8)
for k, r in enumerate(items):
    im = cv2.imread(f"{OUT}/overlays/{r['file']}")
    if im is None:
        continue
    im = cv2.resize(im, (tw, th))
    cv2.putText(im, f"{r['file'][-9:-4]} n{r['n_det']}", (3, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    sheet[(k//6)*th:(k//6+1)*th, (k%6)*tw:(k%6+1)*tw] = im
cv2.imwrite(f'{OUT}/contact_villas.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

counts = [r['n_det'] for r in report]
print(f'frames procesados: {len(report)}')
print(f'detecciones total: {sum(counts)} | media: {np.mean(counts):.1f} | min: {min(counts)} | max: {max(counts)}')
print(f'frames con 0 det: {sum(1 for c in counts if c==0)}')
