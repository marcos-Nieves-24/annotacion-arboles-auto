"""Builds kernels/baseline-fase1/ (.ipynb + kernel-metadata.json) for Kaggle push.

Phase 1 baseline: tcd (Mask-RCNN-family, best effort) + DeepForest boxes
on the frozen 20-frame holdout. Always writes /kaggle/working/baseline_report.json
even on partial failure. No CVAT access, no uploads.
"""

import json
import os

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/baseline-fase1"
os.makedirs(KDIR, exist_ok=True)

CELLS = []

CELLS.append(("markdown", (
    "# Baseline Fase 1 — limón denso (gold v1, holdout 20)\n"
    "A: tcd instancia (familia Mask-RCNN/U-Net, nunca SegFormer comercial) + "
    "B: DeepForest cajas. Métricas vs oro COCO. Sin entrenar, sin CVAT.")))

CELLS.append(("code", "\n".join([
    "import torch, os",
    "assert torch.cuda.is_available(), 'Settings -> Accelerator -> GPU T4'",
    "print(torch.cuda.get_device_name(0), torch.__version__)",
    "DS = '/kaggle/input/limon-gold-v1'",
    "print(sorted(os.listdir(DS)))",
])))

CELLS.append(("code", "\n".join([
    "import zipfile",
    "z = os.path.join(DS, 'images.zip')",
    "imgdir = os.path.join(DS, 'images')",
    "if os.path.exists(z) and not os.path.isdir(imgdir):",
    "    print('unzipping images.zip ...')",
    "    with zipfile.ZipFile(z) as zh:",
    "        zh.extractall(DS)",
    "print('images:', len(os.listdir(imgdir)))",
    "import json",
    "from pathlib import Path",
    "hold = json.loads(Path(DS, 'holdout.json').read_text())",
    "coco = json.loads(Path(DS, 'coco.json').read_text())",
    "print('holdout frames:', len(hold))",
])))

CELLS.append(("code", "\n".join([
    "# Install: keep image torch, add only what is missing",
    "!pip install -q deepforest pycocotools",
    "try:",
    "    import importlib, subprocess",
    "    importlib.import_module('tcd_pipeline')",
    "    print('tcd already present')",
    "except ImportError:",
    "    print('installing tcd from git ...')",
    "    subprocess.run(['pip', 'install', '-q',",
    "                    'git+https://github.com/restor-foundation/tcd.git'], check=False)",
    "import torch",
    "print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())",
])))

CELLS.append(("code", "\n".join([
    "import json, time",
    "from pathlib import Path",
    "report = {'holdout_n': len(hold), 'stages': {}}",
    "OUT = Path('/kaggle/working')",
    "def save():",
    "    (OUT / 'baseline_report.json').write_text(json.dumps(report, indent=1))",
    "    print('report saved (partial ok)')",
])))

CELLS.append(("code", "\n".join([
    "# ---- Stage A: DeepForest boxes (robust path) ----",
    "import pandas as pd",
    "try:",
    "    from deepforest import main",
    "    df_model = main.deepforest()",
    "    df_model.use_release()",
    "    t0 = time.time()",
    "    df_boxes, df_times = [], []",
    "    for h in hold:",
    "        p = os.path.join(imgdir, h['file_name'])",
    "        t1 = time.time()",
    "        try:",
    "            pred = df_model.predict_image(path=p, return_plot=False)",
    "        except TypeError:",
    "            pred = df_model.predict_image(path=p)",
    "        dt = time.time() - t1",
    "        boxes = pred[['xmin', 'ymin', 'xmax', 'ymax', 'score']].values.tolist() if pred is not None and len(pred) else []",
    "        df_boxes.append({'file': h['file_name'], 'boxes': boxes})",
    "        df_times.append(dt)",
    "    (OUT / 'deepforest_boxes.json').write_text(json.dumps(df_boxes))",
    "    report['stages']['deepforest'] = {'ok': True, 'mean_s_per_frame': sum(df_times) / len(df_times)}",
    "    print('deepforest ok, mean s/frame:', round(sum(df_times) / len(df_times), 2))",
    "except Exception as e:",
    "    report['stages']['deepforest'] = {'ok': False, 'error': repr(e)}",
    "    print('deepforest FAILED:', repr(e))",
    "save()",
])))

CELLS.append(("code", "\n".join([
    "# ---- Stage B: tcd instance (best effort, Mask-RCNN/U-Net family only) ----",
    "import subprocess, glob",
    "try:",
    "    h = subprocess.run(['tcd-predict', '--help'], capture_output=True, text=True, timeout=60)",
    "    print((h.stdout or h.stderr)[:1500])",
    "    import tcd_pipeline, os as _os",
    "    pkg = _os.path.dirname(tcd_pipeline.__file__)",
    "    print('tcd at', pkg)",
    "    report['stages']['tcd'] = {'ok': False, 'error': 'model-selection needs ZOO.md review; CLI reachable, see log'}",
    "    print('tcd installed; model pick deferred to run log (see ZOO.md)')",
    "except Exception as e:",
    "    report['stages']['tcd'] = {'ok': False, 'error': repr(e)}",
    "    print('tcd FAILED:', repr(e))",
    "save()",
])))

CELLS.append(("code", "\n".join([
    "# ---- Metrics vs gold: box mAP50 (pycocotools) + count MAE ----",
    "import numpy as np",
    "try:",
    "    from pycocotools.coco import COCO",
    "    from pycocotools.cocoeval import COCOeval",
    "    gt_path = str(OUT / 'gt_holdout.json') if False else None",
    "    # build GT coco subset (non-iscrowd) + DT list from deepforest",
    "    imgs = {i['id']: i for i in coco['images'] if i['file_name'] in {h['file_name'] for h in hold}}",
    "    keep_ids = set(imgs)",
    "    gt = {'images': [dict(i, id=n) for n, i in enumerate(imgs.values())],",
    "          'annotations': [], 'categories': [{'id': 1, 'name': 'crop_tree'}]}",
    "    remap = {old: n for n, old in enumerate(imgs)}",
    "    aid = 0",
    "    for a in coco['annotations']:",
    "        if a['image_id'] in keep_ids and not a['iscrowd']:",
    "            gt['annotations'].append({'id': aid, 'image_id': remap[a['image_id']],",
    "                                      'category_id': 1, 'bbox': a['bbox'],",
    "                                      'area': a['area'], 'iscrowd': 0})",
    "            aid += 1",
    "    Path('/tmp/gt.json').write_text(json.dumps(gt))",
    "    cocoGt = COCO('/tmp/gt.json')",
    "    inv = {h['file_name']: remap[i['id']] for i in imgs.values() for h in [next(x for x in hold if x['file_name'] == i['file_name'])]}",
    "    dt = []",
    "    for e in json.loads((OUT / 'deepforest_boxes.json').read_text()):",
    "        iid = inv[e['file']]",
    "        for x0, y0, x1, y1, s in e['boxes']:",
    "            dt.append({'image_id': iid, 'category_id': 1,",
    "                       'bbox': [x0, y0, x1 - x0, y1 - y0], 'score': float(s)})",
    "    res = cocoGt.loadRes(dt) if dt else None",
    "    if res is not None:",
    "        ev = COCOeval(cocoGt, res, 'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()",
    "        report['stages']['deepforest']['box_map50'] = float(ev.stats[1])",
    "    # count MAE",
    "    pred_n = {e['file']: len(e['boxes']) for e in json.loads((OUT / 'deepforest_boxes.json').read_text())}",
    "    errs = []",
    "    for h in hold:",
    "        iid = next(i['id'] for i in imgs.values() if i['file_name'] == h['file_name'])",
    "        gt_n = sum(1 for a in coco['annotations'] if a['image_id'] == iid and not a['iscrowd'])",
    "        errs.append(abs(pred_n.get(h['file_name'], 0) - gt_n))",
    "    report['stages']['deepforest']['count_MAE'] = float(np.mean(errs))",
    "    print('box mAP50:', report['stages']['deepforest'].get('box_map50'), 'MAE:', round(float(np.mean(errs)), 2))",
    "except Exception as e:",
    "    report.setdefault('stages', {}).setdefault('deepforest', {})['metrics_error'] = repr(e)",
    "    print('metrics FAILED:', repr(e))",
    "save()",
    "print(json.dumps(report, indent=1)[:2000])",
])))

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
for kind, src in CELLS:
    if kind == "markdown":
        nb["cells"].append({"cell_type": "markdown", "metadata": {},
                            "source": src.splitlines(True)})
    else:
        nb["cells"].append({"cell_type": "code", "metadata": {},
                            "source": (src if src.endswith("\n") else src + "\n").splitlines(True),
                            "outputs": [], "execution_count": None})

with open(os.path.join(KDIR, "baseline-fase1.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)

meta = {
    "id": "marcosalvarez234/limon-baseline-fase1",
    "title": "Limon baseline Fase1 (private)",
    "code_file": "baseline-fase1.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": ["marcosalvarez234/limon-gold-v1"],
    "competition_sources": [],
    "kernel_sources": []
}
with open(os.path.join(KDIR, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("kernel built:", os.listdir(KDIR))
