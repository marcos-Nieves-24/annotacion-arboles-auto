# %% [markdown]
# # StarDist en frames 262-300 de Villas (24 imagenes disponibles)
# Mismo pipeline que limon-stardist-261-270: pesos limon-stardist-v1,
# normalize(1, 99.8), barrido NMS 0.02/0.05/0.10, export de poligonos CVAT.
# Faltan los frames 280-294 (imagenes 00790-00804): no estan disponibles.

# %%
import subprocess, sys
for pkg in ['stardist', 'csbdeep']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)
import glob, json, math, os, shutil
import cv2, numpy as np
from PIL import Image
from csbdeep.utils import normalize
from stardist.models import StarDist2D

OUT = '/kaggle/working'
cfg = glob.glob('/kaggle/input/**/config.json', recursive=True)
wf = glob.glob('/kaggle/input/**/weights_best.h5', recursive=True)
assert cfg and wf, f'pesos no encontrados cfg={cfg} w={wf}'
wroot = os.path.dirname(wf[0])
md = os.path.join(OUT, 'model', 'limon_stardist_v1')
os.makedirs(md, exist_ok=True)
for f in ('config.json', 'weights_best.h5', 'thresholds.json'):
    p = os.path.join(wroot, f)
    if os.path.exists(p):
        shutil.copyfile(p, os.path.join(md, f))
model = StarDist2D(None, name='limon_stardist_v1', basedir=os.path.join(OUT, 'model'))
prob = float(getattr(model.thresholds, 'prob', 0.59))
print('pesos cargados, prob', prob, flush=True)

# %%
# imagenes del dataset villas-262-300 (24 disponibles)
imgs = sorted(glob.glob('/kaggle/input/**/images/*.png', recursive=True)
              + glob.glob('/kaggle/input/**/*.png', recursive=True))
imgs = [p for p in imgs if 'Limon_LasVillas_0' in os.path.basename(p)]
# dedupe por nombre (los dos globs pueden repetir la misma ruta)
seen, unique = set(), []
for p in imgs:
    b = os.path.basename(p)
    if b not in seen:
        seen.add(b)
        unique.append(p)
imgs = sorted(unique, key=lambda p: os.path.basename(p))
print('imagenes', len(imgs), [os.path.basename(p) for p in imgs], flush=True)
assert len(imgs) == 24, f'esperaba 24 imagenes, hay {len(imgs)}'

def frame_of(path):
    num = int(os.path.basename(path).split('_')[-1].replace('.png', ''))
    return num - 510

# %%
def pred_mask(img, nms):
    out = model.predict_instances(img, prob_thresh=prob, nms_thresh=nms)
    return out[0] if out[0].ndim == 2 else out[1]

def masks_from(lab):
    return [lab == v for v in np.unique(lab) if v > 0]

PALETTE = [(255,60,60),(60,160,255),(255,200,40),(130,255,90),(220,100,255),
           (255,130,60),(60,230,200),(255,90,160),(150,150,255),(200,255,60),
           (255,170,200),(90,200,130),(180,120,255),(255,220,120),(100,255,180),(240,120,120)]

report = {}
for nms in (0.02, 0.05, 0.10):
    res = []
    for p in imgs:
        name = os.path.basename(p)
        raw = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
        x = normalize(raw, 1, 99.8, axis=(0,1))
        lab = pred_mask(x, nms)
        ms = masks_from(lab)
        union = np.zeros((512,512), np.uint8)
        for m in ms:
            union |= m.astype(np.uint8)
        ncomp = cv2.connectedComponents(union, 8)[0] - 1
        ov = 0
        for i in range(len(ms)):
            for j in range(i+1, len(ms)):
                u = np.count_nonzero(ms[i] | ms[j])
                if u and np.count_nonzero(ms[i] & ms[j]) / u > 0.05:
                    ov += 1
        res.append({'frame': frame_of(p), 'image': name,
                    'n_mascaras': len(ms), 'n_componentes': ncomp, 'pares_solape': ov})
        vis = raw.copy()
        for k, m in enumerate(ms):
            col = PALETTE[k % len(PALETTE)]
            vis[m] = (0.45*vis[m] + 0.55*np.array(col)).astype(np.uint8)
            cnts,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col, 1)
        Image.fromarray(vis).save(os.path.join(OUT, f'nms{nms}_{name}'))
    report[f'nms_{nms}'] = res
    print('nms', nms, 'mascaras', sum(r['n_mascaras'] for r in res),
          'componentes', sum(r['n_componentes'] for r in res),
          'solapes', sum(r['pares_solape'] for r in res), flush=True)
json.dump(report, open(os.path.join(OUT, 'stardist_262_300_report.json'), 'w'), indent=1)

# %%
# export CVAT: un JSON por frame a nms 0.02 + uno combinado
LABEL_ID = 8155070
LABEL = 'crop_tree'
all_shapes = []
per_frame = {}
for p in imgs:
    frame = frame_of(p)
    raw = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
    x = normalize(raw, 1, 99.8, axis=(0,1))
    lab = pred_mask(x, 0.02)
    shapes = []
    for m in masks_from(lab):
        cnts,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(cnt, 0.005*cv2.arcLength(cnt, True), True).ravel()
        if len(poly) < 6:
            continue
        shapes.append({
            'type': 'polygon', 'label_id': LABEL_ID, 'label': LABEL,
            'frame': frame, 'group': 0, 'source': 'auto',
            'points': [round(float(v), 2) for v in poly],
            'occluded': False, 'outside': False, 'z_order': 0,
        })
    wrapper = {'version': 2.0, 'tags': [], 'shapes': shapes, 'tracks': [], 'intervals': []}
    json.dump(wrapper, open(os.path.join(OUT, f'cvat_{frame}_stardist.json'), 'w'), indent=1)
    per_frame[str(frame)] = {'image': os.path.basename(p), 'n_poligonos': len(shapes)}
    all_shapes.extend(shapes)
    print('frame', frame, 'poligonos', len(shapes), flush=True)

combined = {'version': 2.0, 'tags': [], 'shapes': all_shapes, 'tracks': [], 'intervals': []}
json.dump(combined, open(os.path.join(OUT, 'cvat_262_300_stardist.json'), 'w'), indent=1)
manifest = {
    'pipeline': 'StarDist2D (pesos limon-stardist-v1) + NMS 0.02',
    'kernel': 'marcosalvarez234/limon-stardist-262-300',
    'model_source': 'dataset marcosalvarez234/limon-stardist-v1-weights',
    'images_source': 'dataset marcosalvarez234/villas-262-300 (24 de 39)',
    'missing_frames': '280-294 (imagenes 00790-00804)',
    'label_id': LABEL_ID, 'label': LABEL,
    'prob_thresh': prob, 'nms_export': 0.02,
    'n_frames': len(imgs), 'n_poligonos_total': len(all_shapes),
    'cvat_upload': 'NO realizar: requiere autorizacion explicita por lote',
    'per_frame': per_frame,
}
json.dump(manifest, open(os.path.join(OUT, 'manifest_262_300.json'), 'w'), indent=1)
print('TOTAL poligonos', len(all_shapes), 'en', len(imgs), 'frames', flush=True)
