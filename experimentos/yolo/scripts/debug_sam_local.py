"""Debug SAM branch locally: why does it lose detections?"""
import sys, json, os, glob
import numpy as np, cv2, torch
sys.path.insert(0, 'scripts')
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

ROOT = '/home/biohacker24/annotacion-arboles-auto'
IMG = glob.glob('/tmp/user/1000/opencode/vc/**/images', recursive=True)[0]
CKPT = '/tmp/user/1000/opencode/sam_vit_b_01ec64.pth'
C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))
D = json.load(open(f'{ROOT}/data/gold_v2/dev.json'))
byid = {i['id']: i['file_name'] for i in C['images']}

if not os.path.exists(CKPT):
    print('SAM ckpt no disponible local, saltando')
    sys.exit(0)

sam = sam_model_registry['vit_b'](checkpoint=CKPT).to('cpu')
pred = SamPredictor(sam)
m = YOLO('/tmp/user/1000/opencode/wm/model_final.pt') if os.path.exists(
    '/tmp/user/1000/opencode/wm/model_final.pt') else None
print('modelo:', bool(m))
if not m:
    sys.exit(0)

AREAS = {'villas': (882, 39224), 'magdala': (410, 18240)}
for iid in D['A'][:3]:
    fn = byid[iid]
    img = cv2.cvtColor(cv2.imread(os.path.join(IMG, fn)), cv2.COLOR_BGR2RGB)
    r = m.predict(img, imgsz=640, conf=0.5, iou=0.35, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    farm = 'villas' if 'Villas' in fn else 'magdala'
    pred.set_image(img)
    kept = 0
    for b in boxes:
        sc = 512.0 / 640.0
        bb = tuple(v * sc for v in b)
        masks, scores, _ = pred.predict(box=np.array(bb, np.float32), multimask_output=False)
        mm = masks[0]
        a = int(mm.sum())
        ctr = (np.where(mm)[1].mean(), np.where(mm)[0].mean())
        inbox = bb[0] <= ctr[0] <= bb[2] and bb[1] <= ctr[1] <= bb[3]
        inarea = AREAS[farm][0] <= a <= AREAS[farm][1]
        print(f'  box640={tuple(round(v) for v in b)} box512={tuple(round(v) for v in bb)} '
              f'area={a} inarea={inarea} ctr={tuple(round(v) for v in ctr)} inbox={inbox} '
              f'ioscore={float(scores[0]):.2f}')
        if inarea and inbox:
            kept += 1
    print(fn, 'boxes:', len(boxes), 'kept:', kept)
