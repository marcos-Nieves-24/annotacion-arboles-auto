"""Full local YOLO+SAM match to compare with Kaggle T2 numbers."""
import sys, json, os, glob
import numpy as np, cv2
sys.path.insert(0, 'scripts')
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

ROOT = '/home/biohacker24/annotacion-arboles-auto'
IMG = glob.glob('/tmp/user/1000/opencode/vc/**/images', recursive=True)[0]
CKPT = '/tmp/user/1000/opencode/sam_vit_b_01ec64.pth'
C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
byid = {i['id']: i['file_name'] for i in C['images']}
model = YOLO('/tmp/user/1000/opencode/wm/model_final.pt')
sam = sam_model_registry['vit_b'](checkpoint=CKPT).to('cpu')
pred = SamPredictor(sam)
AREAS = {'villas': (882, 39224), 'magdala': (410, 18240)}

def match(refs, preds, thr=0.5):
    pairs = []
    for gi, g in enumerate(refs):
        for pi, p in enumerate(preds):
            it = np.logical_and(g, p).sum(); un = np.logical_or(g, p).sum()
            if un and it / un >= thr: pairs.append((it / un, gi, pi))
    pairs.sort(reverse=True); ug, up, tp = set(), set(), 0
    for _, gi, pi in pairs:
        if gi not in ug and pi not in up: ug.add(gi); up.add(pi); tp += 1
    return tp

for label, ids in [('A', D['A']), ('B', D['B'])]:
    ty = ts = fy = fs = 0
    for iid in ids:
        fn = byid[iid]; p = os.path.join(IMG, fn)
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        farm = 'villas' if 'Villas' in fn else 'magdala'
        r = model.predict(img, imgsz=640, conf=0.5, iou=0.35, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        yolo_masks = []
        if r.masks is not None and len(r.masks):
            md = r.masks.data.float().cpu().numpy()
            for k in range(md.shape[0]):
                mm = md[k] > 0.5
                if mm.shape != (512, 512): mm = cv2.resize(mm.astype('uint8'), (512, 512)) > 0.5
                yolo_masks.append(mm)
        pred.set_image(img)
        sam_masks = []
        for b in boxes:
            sc = 512.0 / 640.0; PAD=15; bb = tuple([max(0,b[0]*sc-PAD),max(0,b[1]*sc-PAD),min(512,b[2]*sc+PAD),min(512,b[3]*sc+PAD)])
            masks, _, _ = pred.predict(box=np.array(bb, np.float32), multimask_output=False)
            mm = masks[0]; a = int(mm.sum()); lo, hi = AREAS[farm]
            ctr = (np.where(mm)[1].mean(), np.where(mm)[0].mean())
            inbox = bb[0] <= ctr[0] <= bb[2] and bb[1] <= ctr[1] <= bb[3]
            if lo <= a <= hi and inbox: sam_masks.append(mm)
        refs = []
        for a in C['annotations']:
            if a['image_id'] == iid and not a['iscrowd'] and a.get('segmentation'):
                pts = np.array(a['segmentation'][0], float).reshape(-1, 2).astype(int)
                mm = np.zeros((512, 512), np.uint8); cv2.fillPoly(mm, [pts], 1)
                refs.append(mm.astype(bool))
        ty += match(refs, yolo_masks); fy += len(yolo_masks) - match(refs, yolo_masks)
        ts += match(refs, sam_masks); fs += len(sam_masks) - match(refs, sam_masks)
    print(f'{label}: YOLO tp={ty} fp={fy} | SAM tp={ts} fp={fs}')
