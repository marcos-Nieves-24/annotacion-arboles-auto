"""Contact sheet of rescue test: green=gold crown covered, red=window positive,
magenta=gold crown NOT covered. For visual inspection of the 8 audited frames.
"""

import json, os, glob
import numpy as np, cv2, torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

ROOT = '/home/biohacker24/annotacion-arboles-auto'
IMG = glob.glob('/tmp/user/1000/opencode/vc/**/images', recursive=True)[0]
OUT = f'{ROOT}/data/fase4/rescue_test/visual'
os.makedirs(OUT, exist_ok=True)
W = '/tmp/user/1000/opencode/texm/model.pth'
C = json.load(open(f'{ROOT}/data/gold_v2/coco.json'))

AUDIT = {
    'Limon_LasVillas_00509.png': 4, 'Limon_LasVillas_00196.png': 2,
    'Limon_LasVillas_00499.png': 12, 'Limon_LasVillas_00563.png': 5,
    'Limon_LasVillas_00519.png': 2, 'Limon_Magdala_00626.png': 2,
    'Limon_Magdala_00866.png': 13, 'Limon_Magdala_00946.png': 4,
}

m = models.resnet18(weights=None)
m.fc = nn.Linear(m.fc.in_features, 1)
m.load_state_dict(torch.load(W, map_location='cpu'))
m.eval()
tf = transforms.Compose([
    transforms.Resize((640, 640)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

sheets = []
for fn in AUDIT:
    p = os.path.join(IMG, fn)
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pos = []
    for y in range(0, h - 256 + 1, 128):
        for x in range(0, w - 256 + 1, 128):
            t = tf(Image.fromarray(img[y:y+256, x:x+256])).unsqueeze(0)
            with torch.no_grad():
                pr = torch.sigmoid(m(t)).item()
            if pr >= 0.5:
                pos.append((x, y, x+256, y+256))
                cv2.rectangle(vis, (x, y), (x+256, y+256), (0, 0, 255), 1)
    iid = next(i['id'] for i in C['images'] if i['file_name'] == fn)
    covered = 0
    total = 0
    for a in C['annotations']:
        if a['image_id'] != iid or a['iscrowd'] or not a.get('segmentation'):
            continue
        pts = np.array(a['segmentation'][0], float).reshape(-1, 2).astype(int)
        cx, cy = pts.mean(axis=0).astype(int)
        total += 1
        hit = any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in pos)
        color = (0, 255, 0) if hit else (255, 0, 255)
        if hit:
            covered += 1
        cv2.circle(vis, (cx, cy), 6, color, -1)
    cv2.putText(vis, f'{fn[-9:-4]} cov {covered}/{total} win {len(pos)}',
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(f'{OUT}/{fn}', vis)
    sheets.append((fn, cv2.resize(vis, (384, 384)), covered, total))

tw, th = 384, 384
rows = (len(sheets) + 3) // 4
sheet = np.full((rows * th, 4 * tw, 3), 40, np.uint8)
for k, (fn, im, cov, tot) in enumerate(sheets):
    sheet[(k//4)*th:(k//4+1)*th, (k%4)*tw:(k%4+1)*tw] = im
cv2.imwrite(f'{OUT}/contact_rescue.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print('contact_rescue.jpg guardado')
