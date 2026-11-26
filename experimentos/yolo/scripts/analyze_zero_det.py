"""Analyze the 21 zero-detection frames from Villas 270-400 inference.

For each, run YOLO at very low conf to see if the model 'sees' the crowns
but below threshold, or truly misses them. Produces a report + contact sheet.
"""
import json, os, glob
import numpy as np, cv2
from ultralytics import YOLO

ROOT = '/home/biohacker24/annotacion-arboles-auto'
D = f'{ROOT}/data/fase4/infer_t1b_villas270_400'
IMG = f'{ROOT}/data/r1/images/villas'
W = '/tmp/user/1000/opencode/wm/model_final.pt'
OUT = f'{ROOT}/data/fase4/infer_t1b_villas270_400/zero_det_analysis'
os.makedirs(f'{OUT}/overlays', exist_ok=True)

report = json.load(open(f'{D}/report.json'))
zeros = [r for r in report['per_frame'] if r['n_det'] == 0]

model = YOLO(W)
rows = []
for r in zeros:
    fn = r['file']
    p = os.path.join(IMG, fn)
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    # very low conf to see latent detections
    res = model.predict(img, imgsz=640, conf=0.01, iou=0.35, verbose=False)[0]
    n01 = len(res.boxes) if res.boxes is not None else 0
    confs = sorted([round(float(c), 3) for c in res.boxes.conf.tolist()], reverse=True) if n01 else []
    rows.append({'file': fn, 'n_conf025': 0, 'n_conf001': n01,
                 'max_conf': confs[0] if confs else None,
                 'confs': confs[:6]})
    print(f"{fn[-9:-4]}: conf0.01={n01} max={confs[0] if confs else '-'}")

json.dump(rows, open(f'{OUT}/report.json', 'w'), indent=1)

# contact sheet at conf 0.01 with boxes
tw = th = 240
cols = 4
sheets = []
for k, row in enumerate(rows):
    fn = row['file']
    img = cv2.cvtColor(cv2.imread(os.path.join(IMG, fn)), cv2.COLOR_BGR2RGB)
    r = model.predict(img, imgsz=640, conf=0.01, iou=0.35, verbose=False)[0]
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if r.boxes is not None:
        for bb in r.boxes.xyxy.tolist():
            x0, y0, x1, y1 = [int(v) for v in bb]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 1)
    cv2.putText(vis, f"{fn[-9:-4]} c001={row['n_conf001']}", (3, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.imwrite(f"{OUT}/overlays/{fn}", vis)
    sheets.append((fn, cv2.resize(vis, (tw, th))))

for bi in range(0, len(sheets), 12):
    chunk = sheets[bi:bi+12]
    rr = (len(chunk) + cols - 1) // cols
    sheet = np.full((rr * th, cols * tw, 3), 40, np.uint8)
    for k, (fn, im) in enumerate(chunk):
        sheet[(k//cols)*th:(k//cols+1)*th, (k%cols)*tw:(k%cols+1)*tw] = im
    cv2.imwrite(f'{OUT}/contact_{bi//12+1:02d}.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

print(f'\nframes 0-det: {len(rows)}')
print(f'con det latente a conf0.01: {sum(1 for r in rows if r["n_conf001"]>0)}')
print(f'sin nada: {sum(1 for r in rows if r["n_conf001"]==0)}')
