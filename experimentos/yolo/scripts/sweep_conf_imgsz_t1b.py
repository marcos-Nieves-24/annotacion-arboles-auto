"""Sweep conf x imgsz on dev for T1b, with the unified instance-F1 metric.

Same weight, same dev frames, same matcher. Read-only, local CPU.
Output: data/fase4/sweep_conf_imgsz.json
"""

import json, os, glob
import numpy as np, cv2
from ultralytics import YOLO

ROOT = '/home/biohacker24/annotacion-arboles-auto'
IMG = glob.glob('/tmp/user/1000/opencode/vc/**/images', recursive=True)[0]
W = '/tmp/user/1000/opencode/wm/model_final.pt'
OUT = f'{ROOT}/data/fase4/sweep_conf_imgsz.json'
C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
H = json.load(open(f'{ROOT}/data/gold_v2/holdout.json'))
hold = set(H['holdout'] if isinstance(H, dict) else [h['file_name'] for h in H])
byid = {i['id']: i['file_name'] for i in C['images']}
dev_names = sorted({byid[i] for i in D['A'] + D['B'] if i in byid} - hold)

# preload images + references once
data = []
for fn in dev_names:
    p = os.path.join(IMG, fn)
    if not os.path.exists(p):
        continue
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    iid = next((i['id'] for i in C['images'] if i['file_name'] == fn), None)
    refs = []
    for a in C['annotations']:
        if a['image_id'] == iid and not a['iscrowd'] and a.get('segmentation'):
            pts = np.array(a['segmentation'][0], float).reshape(-1, 2).astype(int)
            m = np.zeros((512, 512), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            refs.append(m.astype(bool))
    data.append((fn, img, refs))

model = YOLO(W)
results = {}
for imgsz in (512, 640):
    for conf in (0.25, 0.35, 0.5, 0.65):
        tp = fp = fn_ = 0
        for _, img, refs in data:
            r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.35, verbose=False)[0]
            preds = []
            if r.masks is not None and len(r.masks):
                md = r.masks.data.float().cpu().numpy()
                for k in range(md.shape[0]):
                    m = md[k] > 0.5
                    if m.shape != (512, 512):
                        m = cv2.resize(m.astype('uint8'), (512, 512)) > 0.5
                    preds.append(m)
            pairs = []
            for gi, g in enumerate(refs):
                for pi, pm in enumerate(preds):
                    it = np.logical_and(g, pm).sum()
                    un = np.logical_or(g, pm).sum()
                    if un and it / un >= 0.5:
                        pairs.append((it / un, gi, pi))
            pairs.sort(reverse=True)
            ug, up = set(), set()
            for _, gi, pi in pairs:
                if gi not in ug and pi not in up:
                    ug.add(gi); up.add(pi); tp += 1
            fp += len(preds) - len(up)
            fn_ += len(refs) - len(ug)
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn_) if tp + fn_ else 0
        F1 = 2 * P * R / (P + R) if P + R else 0
        key = f"sz{imgsz}_conf{conf}"
        results[key] = {"P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
                        "tp": tp, "fp": fp, "fn": fn_}
        print(key, results[key], flush=True)
json.dump(results, open(OUT, "w"), indent=1)
print("saved", OUT)
