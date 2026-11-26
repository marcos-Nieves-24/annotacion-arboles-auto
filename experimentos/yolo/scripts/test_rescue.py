"""Rescue test: slide-window ResNet-18 texture classifier on the 9 audited frames.

For each frame with 0 YOLO detections but real crowns (user audit), slide a
640 window with stride 320, classify each window, NMS the positive windows,
and compare the rescued crown count with the user's visual count and gold.

Output: data/fase4/rescue_test/{overlays,report.json}
"""

import json, os, glob, re
import numpy as np, cv2, torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

ROOT = '/home/biohacker24/annotacion-arboles-auto'
OUT = f'{ROOT}/data/fase4/rescue_test'
os.makedirs(f'{OUT}/overlays', exist_ok=True)
W = '/tmp/user/1000/opencode/texm/model.pth'
IMG = glob.glob('/tmp/user/1000/opencode/vc/**/images', recursive=True)[0]
SIZE = 256
STRIDE = 128
THR = 0.5

# user audit: frame -> approximate real crowns
AUDIT = {
    'Limon_LasVillas_00509.png': 4,
    'Limon_LasVillas_00196.png': 2,
    'Limon_LasVillas_00499.png': 12,
    'Limon_LasVillas_00563.png': 5,
    'Limon_LasVillas_00519.png': 2,
    'Limon_Magdala_00626.png': 2,
    'Limon_Magdala_00866.png': 13,
    'Limon_Magdala_00946.png': 4,
}
# 00408 = small crowns, no count given

C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))

# load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(W, map_location='cpu'))
model.eval()
tf = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def gold_count(fn):
    iid = next((i['id'] for i in C['images'] if i['file_name'] == fn), None)
    if iid is None:
        return 0
    return sum(1 for a in C['annotations'] if a['image_id'] == iid and not a['iscrowd'])


def nms_boxes(boxes, probs, iou_thr=0.3):
    order = np.argsort(-probs)
    keep = []
    for idx in order:
        b = boxes[idx]
        ok = True
        for k in keep:
            kb = boxes[k]
            xx = max(0, min(b[2], kb[2]) - max(b[0], kb[0]))
            yy = max(0, min(b[3], kb[3]) - max(b[1], kb[1]))
            inter = xx * yy
            area = (b[2]-b[0])*(b[3]-b[1]) + (kb[2]-kb[0])*(kb[3]-kb[1]) - inter
            if area > 0 and inter / area > iou_thr:
                ok = False; break
        if ok:
            keep.append(idx)
    return keep


report = []
for fn, user_count in AUDIT.items():
    p = os.path.join(IMG, fn)
    if not os.path.exists(p):
        print('missing', fn); continue
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    boxes, probs = [], []
    for y in range(0, max(1, h - SIZE + 1), STRIDE):
        for x in range(0, max(1, w - SIZE + 1), STRIDE):
            crop = img[y:min(y+SIZE, h), x:min(x+SIZE, w)]
            t = tf(Image.fromarray(crop)).unsqueeze(0)
            with torch.no_grad():
                pr = torch.sigmoid(model(t)).item()
            if pr >= THR:
                boxes.append([x, y, min(x+SIZE, w), min(y+SIZE, h)])
                probs.append(pr)
    keep = nms_boxes(np.array(boxes) if boxes else np.zeros((0,4)),
                     np.array(probs) if probs else np.array([])) if boxes else []
    n_rescued = len(keep)
    g = gold_count(fn)
    # overlay
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for idx in keep:
        x0, y0, x1, y1 = boxes[idx]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(vis, f'{probs[idx]:.2f}', (x0, y0+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(f'{OUT}/overlays/{fn}', vis)
    entry = {'file': fn, 'user_crowns': user_count, 'gold_crowns': g,
             'rescued_windows': n_rescued,
             'recovery_vs_user': round(n_rescued/user_count, 2) if user_count else None}
    report.append(entry)
    print(fn, 'user', user_count, 'gold', g, 'rescued', n_rescued, flush=True)

json.dump(report, open(f'{OUT}/report.json', 'w'), indent=1)
tot_u = sum(r['user_crowns'] for r in report)
tot_r = sum(r['rescued_windows'] for r in report)
print(f'TOTAL: user crowns={tot_u} rescued windows={tot_r} ratio={tot_r/tot_u:.2f}')
