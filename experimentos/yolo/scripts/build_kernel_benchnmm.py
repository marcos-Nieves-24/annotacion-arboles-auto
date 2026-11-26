"""Builds kernels/bench-nmm/ for Kaggle push.

Benchmark on T4: supervision with_nmm (0.05/0.1/0.15/0.2) vs count-merge,
all 331 clean-gold frames, r1 weights, conf 0.5. Saves nmm_bench.json.
"""

import json
import os
import uuid

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/bench-nmm"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# Benchmark NMM vs count-merge (331 oro, T4)\n"
          "Ramas por frame + tabla P/R/F1 máscara. Sin entrenar."))

C.append(("code", "\n".join([
    "import torch, os",
    "print('cuda:', torch.cuda.is_available())",
    "assert torch.cuda.is_available(), 'Settings -> Accelerator -> GPU T4'",
    "GOLD = next(d for d in __import__('glob').glob('/kaggle/input/datasets/*/*') if os.path.exists(os.path.join(d, 'coco.json')))",
    "WDS = next(d for d in __import__('glob').glob('/kaggle/input/datasets/*/*') if os.path.exists(os.path.join(d, 'r1-best.pt')))",
    "print('GOLD =', GOLD)",
    "print('W =', WDS)",
])))

C.append(("code", "\n".join([
    "import zipfile",
    "for ds in (GOLD,):",
    "    for z in sorted(__import__('glob').glob(os.path.join(ds, '*.zip'))):",
    "        print('unzipping', os.path.basename(z))",
    "        with zipfile.ZipFile(z) as zh:",
    "            zh.extractall(ds)",
    "print('gold files:', sorted(os.listdir(GOLD))[:8])",
])))

C.append(("code", "\n".join([
    "import importlib, subprocess",
    "for pkg, spec in [('ultralytics', 'ultralytics'), ('supervision', 'supervision'), ('pycocotools', 'pycocotools'), ('cv2', 'opencv-python-headless')]:",
    "    try: importlib.import_module(pkg); print(pkg, 'present')",
    "    except ImportError:",
    "        print('installing', spec); subprocess.run(['pip', 'install', '-q', spec])",
    "import supervision as sv",
    "print('supervision', sv.__version__)",
])))

C.append(("code", "\n".join([
    "import json, numpy as np, cv2",
    "from pathlib import Path",
    "OUT = Path('/kaggle/working')",
    "coco = json.loads(Path(GOLD, 'coco.json').read_text())",
    "gv = {}",
    "for a in coco['annotations']:",
    "    if not a['iscrowd']: gv.setdefault(a['image_id'], []).append(a['segmentation'])",
    "byname = {i['file_name']: i['id'] for i in coco['images']}",
    "farm_of = {i['file_name']: i['farm'] for i in coco['images']}",
    "AREA = {'villas': (882, 39224), 'magdala': (410, 18240), 'yulima': (767, 34096)}",
    "MED = {'villas': 9798, 'magdala': 6636, 'yulima': 4057}",
    "def rast(pts):",
    "    m = np.zeros((512, 512), np.uint8)",
    "    cv2.fillPoly(m, [np.array(pts, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)",
    "    return m",
    "def miou(a, b):",
    "    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()",
    "    return float(i / u) if u else 0.0",
    "json.dump({'farms': sorted(set(farm_of.values()))}, open('/tmp/x.json', 'w'))",
    "print('gold images:', len(byname))",
])))

C.append(("code", "\n".join([
    "from ultralytics import YOLO",
    "import torch",
    "model = YOLO(os.path.join(WDS, 'r1-best.pt'))",
    "files = sorted(byname.keys())",
    "paths = [os.path.join(GOLD, 'images', f) for f in files]",
    "missing = [p for p in paths if not os.path.exists(p)]",
    "assert not missing, f'missing {len(missing)} images'",
    "res = model.predict(paths, imgsz=640, conf=0.5, iou=0.5, verbose=False)",
    "import pickle",
    "raw = []",
    "for r in res:",
    "    nb = len(r.boxes) if r.boxes is not None else 0",
    "    boxes = r.boxes.xyxy.cpu().numpy() if nb else np.zeros((0, 4))",
    "    confs = r.boxes.conf.cpu().numpy() if nb else np.zeros(0)",
    "    masks = None",
    "    if r.masks is not None and len(r.masks):",
    "        import torch.nn.functional as F",
    "        md = r.masks.data.float()",
    "        if md.shape[1] != 512 or md.shape[2] != 512:",
    "            md = F.interpolate(md.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)",
    "        masks = (md > 0.5).cpu().numpy().astype('uint8')",
    "    raw.append({'boxes': boxes, 'confs': confs, 'masks': masks})",
    "pickle.dump(raw, open('/tmp/raw.pkl', 'wb'))",
    "print('predicted:', len(raw))",
])))

C.append(("code", "\n".join([
    "import pickle",
    "raw = pickle.load(open('/tmp/raw.pkl', 'rb'))",
    "files = sorted(byname.keys())",
    "agg = {}",
    "def score(pred_masks, g):",
    "    used, tp = set(), 0",
    "    for p in pred_masks:",
    "        best, bj = 0.0, -1",
    "        for j, a in enumerate(g):",
    "            if j in used: continue",
    "            v = miou(p, a)",
    "            if v > best: best, bj = v, j",
    "        if best >= 0.5: tp += 1; used.add(bj)",
    "    P = tp / max(len(pred_masks), 1); R = tp / max(len(g), 1)",
    "    return P, R",
    "for idx, fn in enumerate(files):",
    "    farm = farm_of[fn]; amin, amax = AREA[farm]",
    "    g = [rast(s) for s in gv.get(byname[fn], [])]",
    "    d = raw[idx]",
    "    nb = len(d['boxes'])",
    "    base = [d['masks'][k] for k in range(nb) if d['masks'] is not None and amin <= d['masks'][k].sum() <= amax]",
    "    # branch A: count-merge",
    "    union = np.logical_or.reduce(base).astype(np.uint8) if base else np.zeros((512, 512), np.uint8)",
    "    exp = max(1, int(round(union.sum() / MED[farm])))",
    "    cur = [m.astype(bool) for m in base]",
    "    guard = 0",
    "    while len(cur) > int(exp * 1.1) + 1 and guard < 200:",
    "        guard += 1",
    "        best, bi, bj = 0, -1, -1",
    "        for i in range(len(cur)):",
    "            di = cv2.dilate(cur[i].astype(np.uint8), np.ones((3, 3), np.uint8))",
    "            for j in range(i + 1, len(cur)):",
    "                sh = int(np.logical_and(di, cur[j]).sum())",
    "                if sh > best and (cur[i] | cur[j]).sum() <= amax: best, bi, bj = sh, i, j",
    "        if bi < 0 or best < 8: break",
    "        cur = [m for k, m in enumerate(cur) if k not in (bi, bj)] + [(cur[bi] | cur[bj]).astype(np.uint8)]",
    "    pa, ra = score([m.astype(np.uint8) for m in cur], g)",
    "    agg.setdefault('ours', []).append((pa, ra))",
    "    # branch B: supervision NMM",
    "    try:",
    "        det = sv.Detections(xyxy=np.array(d['boxes']), confidence=np.array(d['confs']), class_id=np.zeros(nb, dtype=int), mask=d['masks'])",
    "        for T in (0.05, 0.1, 0.15, 0.2):",
    "            try:",
    "                mrg = det.with_nmm(threshold=T)",
    "                pm = []",
    "                if mrg.mask is not None:",
    "                    for k in range(len(mrg)):",
    "                        m = mrg.mask[k].astype(np.uint8)",
    "                        if amin <= m.sum() <= amax: pm.append(m)",
    "                pb, rb = score(pm, g)",
    "            except Exception as e:",
    "                pb, rb = -1.0, -1.0",
    "                if idx == 0: print('nmm fail', T, repr(e)[:150])",
    "            agg.setdefault(f'nmm_{T}', []).append((pb, rb))",
    "    except Exception as e:",
    "        if idx == 0: print('det build fail:', repr(e)[:150])",
    "    if (idx + 1) % 50 == 0: print(f'{idx + 1}/331')",
    "out = {}",
    "for k, v in agg.items():",
    "    P = float(np.mean([p for p, _ in v if p >= 0])); R = float(np.mean([r for _, r in v if r >= 0]))",
    "    out[k] = {'P': round(P, 3), 'R': round(R, 3), 'F1': round(2 * P * R / max(P + R, 1e-6), 3), 'n': len(v)}",
    "print(json.dumps(out, indent=1))",
    "json.dump(out, open('/kaggle/working/nmm_bench.json', 'w'), indent=1)",
])))

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {
          "kernelspec": {"display_name": "Python 3", "language": "python",
                         "name": "python3"},
          "language_info": {"name": "python", "version": "3.10.0"}},
      "cells": []}
for kind, src in C:
    cid = uuid.uuid4().hex[:8]
    if kind == "markdown":
        nb["cells"].append({"cell_type": "markdown", "id": cid, "metadata": {},
                            "source": src.splitlines(True)})
    else:
        nb["cells"].append({"cell_type": "code", "id": cid, "metadata": {},
                            "source": (src if src.endswith("\n") else src + "\n").splitlines(True),
                            "outputs": [], "execution_count": None})

with open(os.path.join(KDIR, "bench-nmm.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)

meta = {
    "id": "marcosalvarez234/limon-bench-nmm",
    "title": "Limon Bench Nmm",
    "code_file": "bench-nmm.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": ["marcosalvarez234/limon-gold-v1",
                        "marcosalvarez234/limon-weights-r1"],
    "competition_sources": [],
    "kernel_sources": []
}
with open(os.path.join(KDIR, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("bench kernel built:", len(nb["cells"]), "cells")
