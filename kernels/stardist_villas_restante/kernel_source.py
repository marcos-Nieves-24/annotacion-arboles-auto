# %% [markdown]
# # StarDist — villas-restante con prob_thresh=0.30
# Usa el mapeo REAL frame->imagen de data/raw/villas/meta.json, guardado como
# frame_map.json en el dataset. No asume frame = numero_imagen - 510.

# %%
import subprocess, sys
for pkg in ['stardist', 'csbdeep']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)

import glob, json, os, shutil
import cv2, numpy as np
from PIL import Image
from csbdeep.utils import normalize
from stardist.models import StarDist2D

OUT = '/kaggle/working'
mpath = glob.glob('/kaggle/input/**/frame_map.json', recursive=True)
assert len(mpath) == 1, f'frame_map ambiguo: {mpath}'
fmap = json.load(open(mpath[0]))
imgs = glob.glob('/kaggle/input/**/images/*.png', recursive=True)
byname = {os.path.basename(p): p for p in imgs}
assert all(v in byname for v in fmap.values()), 'falta imagen del mapa'
print('frames', len(fmap), sorted(int(k) for k in fmap), flush=True)

cfg = glob.glob('/kaggle/input/**/config.json', recursive=True)
wf = glob.glob('/kaggle/input/**/weights_best.h5', recursive=True)
assert cfg and wf, f'pesos no encontrados: {cfg} {wf}'
wroot = os.path.dirname(wf[0])
md = os.path.join(OUT, 'model', 'limon_stardist_v1')
os.makedirs(md, exist_ok=True)
for f in ('config.json', 'weights_best.h5', 'thresholds.json'):
    p = os.path.join(wroot, f)
    if os.path.exists(p):
        shutil.copyfile(p, os.path.join(md, f))
model = StarDist2D(None, name='limon_stardist_v1', basedir=os.path.join(OUT, 'model'))
prob = 0.30  # umbral elegido por el usuario
print('prob', prob, flush=True)

# %%
LABEL_ID = 8155070
PALETTE = [(255,60,60),(60,160,255),(255,200,40),(130,255,90),(220,100,255),
           (255,130,60),(60,230,200),(255,90,160),(150,150,255),(200,255,60),
           (255,170,200),(90,200,130),(180,120,255),(255,220,120),(100,255,180),(240,120,120)]

def masks_from(lab):
    return [lab == v for v in np.unique(lab) if v > 0]

report = {}
for frame_str in sorted(fmap, key=int):
    frame = int(frame_str)
    name = fmap[frame_str]
    raw = np.asarray(Image.open(byname[name]).convert('RGB'), dtype=np.uint8)
    x = normalize(raw, 1, 99.8, axis=(0, 1))
    out = model.predict_instances(x, prob_thresh=prob, nms_thresh=0.02)
    lab = out[0] if out[0].ndim == 2 else out[1]
    ms = masks_from(lab)
    union = np.zeros((512, 512), np.uint8)
    for m in ms:
        union |= m.astype(np.uint8)
    ncomp = cv2.connectedComponents(union, 8)[0] - 1
    ov = 0
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            u = np.count_nonzero(ms[i] | ms[j])
            if u and np.count_nonzero(ms[i] & ms[j]) / u > 0.05:
                ov += 1

    shapes = []
    vis = raw.copy()
    for k, m in enumerate(ms):
        col = PALETTE[k % len(PALETTE)]
        vis[m] = (0.45 * vis[m] + 0.55 * np.array(col)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(vis, cnts, -1, col, 1)
        poly = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True).ravel()
        if len(poly) < 6:
            continue
        shapes.append({
            'type': 'polygon', 'label_id': LABEL_ID, 'label': 'crop_tree',
            'frame': frame, 'group': 0, 'source': 'auto',
            'points': [round(float(v), 2) for v in poly],
            'occluded': False, 'outside': False, 'z_order': 0,
        })
    Image.fromarray(vis).save(os.path.join(OUT, f'nms0.02_{name}'))

    wrapper = {'version': 2.0, 'tags': [], 'shapes': shapes, 'tracks': [], 'intervals': []}
    json.dump(wrapper, open(os.path.join(OUT, f'cvat_{frame}_stardist.json'), 'w'), indent=1)
    report[str(frame)] = {'image': name, 'n_mascaras': len(ms),
                          'n_componentes': ncomp, 'pares_solape': ov,
                          'n_poligonos': len(shapes)}
    print('frame', frame, name, 'poligonos', len(shapes), flush=True)

json.dump(report, open(os.path.join(OUT, 'stardist_stardist_villas_restante_report.json'), 'w'), indent=1)
print('TOTAL', sum(v['n_poligonos'] for v in report.values()), flush=True)
