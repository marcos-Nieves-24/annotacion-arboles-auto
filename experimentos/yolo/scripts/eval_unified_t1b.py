"""Unified evaluation of T1b on dev and holdout with the SAME metric function.

Instance P/R/F1 (IoU>=0.5 one-to-one), count MAE, over-split and under-split rate.
Predictions: YOLO masks rescaled to 512. References: gold_v2 COCO non-iscrowd.
Read-only. Local CPU.
"""

import json, os, glob
import numpy as np, cv2
from ultralytics import YOLO

ROOT = '/home/biohacker24/annotacion-arboles-auto'
IMG = glob.glob('/tmp/user/1000/opencode/vc/**/images', recursive=True)[0]
W = '/tmp/user/1000/opencode/wm/model_final.pt'
OUT = f'{ROOT}/data/fase4/eval_unified_t1b.json'

C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
H = json.load(open(f'{ROOT}/data/gold_v2/holdout.json'))
hold = set(H['holdout'] if isinstance(H, dict) else [h['file_name'] for h in H])
byid = {i['id']: i['file_name'] for i in C['images']}
dev_names = {byid[i] for i in D['A'] + D['B'] if i in byid} - hold

model = YOLO(W)


def evaluate(names, label):
    tp = fp = fn = 0
    errs = []
    over = 0   # GT represented by >=2 predicted instances
    under = 0  # predicted instance containing >=2 GT trees
    for frame_name in sorted(names):
        p = os.path.join(IMG, frame_name)
        if not os.path.exists(p):
            continue
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        r = model.predict(img, imgsz=640, conf=0.5, iou=0.35, verbose=False)[0]
        preds = []
        if r.masks is not None and len(r.masks):
            md = r.masks.data.float().cpu().numpy()
            for k in range(md.shape[0]):
                m = md[k] > 0.5
                if m.shape != (512, 512):
                    m = cv2.resize(m.astype('uint8'), (512, 512)) > 0.5
                preds.append(m)
        iid = next((i['id'] for i in C['images'] if i['file_name'] == frame_name), None)
        refs = []
        for a in C['annotations']:
            if a['image_id'] == iid and not a['iscrowd'] and a.get('segmentation'):
                pts = np.array(a['segmentation'][0], float).reshape(-1, 2).astype(int)
                m = np.zeros((512, 512), np.uint8)
                cv2.fillPoly(m, [pts], 1)
                refs.append(m.astype(bool))

        # IoU matrix
        ious = np.zeros((len(refs), len(preds)), np.float32)
        for gi, g in enumerate(refs):
            for pi, pm in enumerate(preds):
                it = np.logical_and(g, pm).sum()
                un = np.logical_or(g, pm).sum()
                if un:
                    ious[gi, pi] = it / un

        # one-to-one matching at IoU >= 0.5, greedy by IoU
        pairs = []
        for gi in range(len(refs)):
            for pi in range(len(preds)):
                if ious[gi, pi] >= 0.5:
                    pairs.append((float(ious[gi, pi]), gi, pi))
        pairs.sort(reverse=True)
        ug, up = set(), set()
        for _, gi, pi in pairs:
            if gi not in ug and pi not in up:
                ug.add(gi); up.add(pi); tp += 1
        # over/under-split evidence (IoU >= 0.25 means overlap but below match threshold)
        for gi in range(len(refs)):
            if (ious[gi] >= 0.25).sum() >= 2:
                over += 1
        for pi in range(len(preds)):
            if (ious[:, pi] >= 0.25).sum() >= 2:
                under += 1

        fp += len(preds) - len(up)
        fn += len(refs) - len(ug)
        errs.append(abs(len(preds) - len(refs)))

    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F1 = 2 * P * R / (P + R) if P + R else 0
    return {"frames": len(names), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(P, 4), "recall": round(R, 4), "f1": round(F1, 4),
            "count_MAE": round(float(np.mean(errs)), 3) if errs else None,
            "over_split_frames": over, "under_split_frames": under}


report = {
    "model": W,
    "metric": "instance P/R/F1 IoU>=0.5 one-to-one; over/under split at IoU>=0.25",
    "dev": evaluate(dev_names, "dev"),
    "holdout": evaluate(hold, "holdout"),
}
json.dump(report, open(OUT, "w"), indent=1)
print(json.dumps(report, indent=1))
