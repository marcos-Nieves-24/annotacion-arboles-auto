"""Replicate T2 dev evaluation locally to isolate where recall is lost."""
import sys, json, os, glob
import numpy as np, cv2
sys.path.insert(0, 'scripts')
from ultralytics import YOLO

ROOT = '/home/biohacker24/annotacion-arboles-auto'
IMG = glob.glob('/tmp/user/1000/opencode/vc/**/images', recursive=True)[0]
C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
byid = {i['id']: i['file_name'] for i in C['images']}
WA = YOLO('/tmp/user/1000/opencode/wtest/runs-t1/splitA/weights/best.pt')
WB = YOLO('/tmp/user/1000/opencode/wtest/runs-t1/splitB/weights/best.pt')

def match(refs, preds, thr=0.5):
    pairs = []
    for gi, g in enumerate(refs):
        for pi, p in enumerate(preds):
            inter = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            if union and inter / union >= thr:
                pairs.append((inter / union, gi, pi))
    pairs.sort(reverse=True)
    ug, up, tp = set(), set(), 0
    for _, gi, pi in pairs:
        if gi not in ug and pi not in up:
            ug.add(gi); up.add(pi); tp += 1
    return tp

for label, ids, model in [('A', D['A'], WA), ('B', D['B'], WB)]:
    tp = fp = fn = 0
    npred = nref = 0
    for iid in ids:
        fnm = byid[iid]
        img = cv2.cvtColor(cv2.imread(os.path.join(IMG, fnm)), cv2.COLOR_BGR2RGB)
        r = model.predict(img, imgsz=640, conf=0.5, iou=0.35, verbose=False)[0]
        preds = []
        if r.masks is not None and len(r.masks):
            md = r.masks.data.float().cpu().numpy()
            for k in range(md.shape[0]):
                m = md[k] > 0.5
                if m.shape != (512, 512):
                    m = cv2.resize(m.astype('uint8'), (512, 512)) > 0.5
                preds.append(m)
        refs = []
        for a in C['annotations']:
            if a['image_id'] == iid and not a['iscrowd'] and a.get('segmentation'):
                pts = np.array(a['segmentation'][0], float).reshape(-1, 2).astype(int)
                m = np.zeros((512, 512), np.uint8)
                cv2.fillPoly(m, [pts], 1)
                refs.append(m.astype(bool))
        t = match(refs, preds)
        tp += t; fp += len(preds) - t; fn += len(refs) - t
        npred += len(preds); nref += len(refs)
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    print(label, 'preds', npred, 'refs', nref, 'tp', tp, 'fp', fp, 'fn', fn,
          'P', round(P, 3), 'R', round(R, 3), 'F1', round(2*P*R/(P+R), 3) if P+R else 0)
