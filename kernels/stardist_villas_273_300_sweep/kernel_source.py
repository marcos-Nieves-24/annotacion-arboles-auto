# %% [markdown]
# # StarDist — barrido de prob_thresh en Villas frames 273-300
# Mismos 28 frames del test 0.55, NMS 0.02, todos los umbrales del barrido.
# Genera un JSON CVAT por frame y umbral + conteos + overlays de 5 frames.

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
byname = {os.path.basename(p): p
          for p in glob.glob('/kaggle/input/**/images/*.png', recursive=True)}
assert all(v in byname for v in fmap.values()), 'falta imagen del mapa'

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
PROBS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, round(float(getattr(model.thresholds, 'prob', 0.59)), 4)]
NMS = 0.02
LABEL_ID = 8155070
SHOW = [273, 280, 288, 294, 300]
print('frames', len(fmap), 'probs', PROBS, 'nms', NMS, flush=True)

# %%
def masks_from(lab):
    return [lab == v for v in np.unique(lab) if v > 0]

report = {'dataset': 'villas-273-300', 'nms': NMS, 'probs': PROBS,
          'n_frames': len(fmap), 'per_prob': {}}
for prob in PROBS:
    tag = f'{prob:.4f}'
    pdir = os.path.join(OUT, tag)
    os.makedirs(pdir, exist_ok=True)
    rows = {}
    for frame_str in sorted(fmap, key=int):
        frame = int(frame_str)
        name = fmap[frame_str]
        raw = np.asarray(Image.open(byname[name]).convert('RGB'), dtype=np.uint8)
        x = normalize(raw, 1, 99.8, axis=(0, 1))
        out = model.predict_instances(x, prob_thresh=prob, nms_thresh=NMS)
        lab = out[0] if out[0].ndim == 2 else out[1]
        ms = masks_from(lab)

        shapes = []
        for m in ms:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            poly = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True).ravel()
            if len(poly) < 6:
                continue
            shapes.append({
                'type': 'polygon', 'label_id': LABEL_ID, 'label': 'crop_tree',
                'frame': frame, 'group': 0, 'source': 'auto',
                'points': [round(float(v), 2) for v in poly],
                'occluded': False, 'outside': False, 'z_order': 0,
            })
        json.dump({'version': 2.0, 'tags': [], 'shapes': shapes,
                   'tracks': [], 'intervals': []},
                  open(os.path.join(pdir, f'cvat_{frame}_stardist.json'), 'w'), indent=1)
        rows[str(frame)] = {'image': name, 'n_poligonos': len(shapes)}

        if frame in SHOW:
            vis = raw.copy()
            palette = [(255,60,60),(60,160,255),(255,200,40),(130,255,90),
                       (220,100,255),(255,130,60),(60,230,200),(255,90,160),
                       (150,150,255),(200,255,60),(255,170,200),(90,200,130),
                       (180,120,255),(255,220,120),(100,255,180),(240,120,120)]
            for k, m in enumerate(ms):
                col = palette[k % len(palette)]
                vis[m] = (0.45 * vis[m] + 0.55 * np.array(col)).astype(np.uint8)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cv2.drawContours(vis, cnts, -1, col, 1)
            Image.fromarray(vis).save(os.path.join(pdir, f'ov_{name}'))

    report['per_prob'][tag] = rows
    print('prob', tag, 'poligonos', sum(v['n_poligonos'] for v in rows.values()),
          flush=True)

json.dump(report, open(os.path.join(OUT, 'stardist_villas_273_300_sweep.json'), 'w'), indent=1)
print('LISTO', flush=True)
