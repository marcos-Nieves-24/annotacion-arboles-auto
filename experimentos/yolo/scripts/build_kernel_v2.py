"""Builds kernels/baseline-fase1 v2: DeepForest-first, tcd optional-last.

v1 failed with KernelWorkerStatus.ERROR (no log via API). Hardening:
- installs only missing packages, re-asserts CUDA torch after pip
- every stage guarded, partial report always saved
- tcd moved to an optional last cell so it can never kill the baseline
"""

import json
import os

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/baseline-fase1"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# Baseline Fase 1 v2 — limón denso (gold v1, holdout 20)\n"
          "DeepForest primero (robusto), tcd al final como opcional. "
          "Todo guardado parcial en `/kaggle/working/baseline_report.json`."))

C.append(("code", "\n".join([
    "import torch, os, glob",
    "print('cuda:', torch.cuda.is_available())",
    "assert torch.cuda.is_available(), 'Settings -> Accelerator -> GPU T4'",
    "print(torch.cuda.get_device_name(0), torch.__version__)",
    "cands = [d for d in glob.glob('/kaggle/input/*') if os.path.isdir(d)]",
    "print('attached inputs:', cands)",
    "DS = None",
    "for root, dirs, files in os.walk('/kaggle/input'):",
    "    if 'holdout.json' in files:",
    "        DS = root; break",
    "    dirs[:] = [d for d in dirs if root.count(os.sep) - '/kaggle/input'.count(os.sep) < 4]",
    "assert DS is not None, f'gold dataset not mounted; inputs={cands}'",
    "print('DS =', DS, sorted(os.listdir(DS)))",
])))

C.append(("code", "\n".join([
    "import zipfile",
    "imgdir = os.path.join(DS, 'images')",
    "try:",
    "    z = os.path.join(DS, 'images.zip')",
    "    if os.path.exists(z) and not os.path.isdir(imgdir):",
    "        print('unzipping images.zip ...')",
    "        with zipfile.ZipFile(z) as zh:",
    "            zh.extractall(DS)",
    "    print('images:', len(os.listdir(imgdir)))",
    "except Exception as e:",
    "    print('UNZIP FAILED:', repr(e))",
])))

C.append(("code", "\n".join([
    "import importlib, subprocess",
    "def ensure(pkg, pip_spec=None):",
    "    try:",
    "        importlib.import_module(pkg)",
    "        print(pkg, 'already present')",
    "        return True",
    "    except ImportError:",
    "        print(f'installing {pip_spec or pkg} ...')",
    "        r = subprocess.run(['pip', 'install', '-q', pip_spec or pkg])",
    "        print('pip rc:', r.returncode)",
    "        return r.returncode == 0",
    "ok_df = ensure('deepforest', 'deepforest')",
    "ok_coco = ensure('pycocotools', 'pycocotools')",
    "import torch",
    "print('torch still cuda:', torch.cuda.is_available(), torch.__version__)",
])))

C.append(("code", "\n".join([
    "import json, time",
    "from pathlib import Path",
    "OUT = Path('/kaggle/working')",
    "report = {'holdout_n': None, 'stages': {}}",
    "def save():",
    "    (OUT / 'baseline_report.json').write_text(json.dumps(report, indent=1))",
    "try:",
    "    hold = json.loads(Path(DS, 'holdout.json').read_text())",
    "    coco = json.loads(Path(DS, 'coco.json').read_text())",
    "    report['holdout_n'] = len(hold)",
    "    print('holdout:', len(hold), '| coco images:', len(coco['images']))",
    "except Exception as e:",
    "    print('LOAD FAILED:', repr(e))",
    "save()",
])))

C.append(("code", "\n".join([
    "ok = True",
    "try:",
    "    from deepforest import main",
    "    m = main.deepforest()",
    "    print('api loaders:', [a for a in dir(m) if 'load' in a.lower() or 'release' in a.lower()])",
    "    if hasattr(m, 'load_model'):",
    "        try: m.load_model(); print('load_model() ok')",
    "        except Exception as e: print('load_model() note:', repr(e)[:200])",
    "    print('deepforest model ready')",
    "except Exception as e:",
    "    ok = False",
    "    report['stages']['deepforest'] = {'ok': False, 'error': repr(e)}",
    "    print('MODEL FAILED:', repr(e))",
    "if ok:",
    "    import pandas as pd, cv2",
    "    boxes_out, times = [], []",
    "    def predict_one(p):",
    "        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGB) if cv2.imread(p, cv2.IMREAD_UNCHANGED).shape[2] == 4 else cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)",
    "        try: return m.predict_image(image=img)",
    "        except TypeError:",
    "            try: return m.predict_image(path=p)",
    "            except TypeError: return m.predict_tile(image=img)",
    "    for h in hold:",
    "        p = os.path.join(imgdir, h['file_name'])",
    "        t1 = time.time()",
    "        try:",
    "            pred = predict_one(p)",
    "            b = pred[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist() if pred is not None and len(pred) else []",
    "            s = pred['score'].values.tolist() if pred is not None and len(pred) and 'score' in pred else [0.5] * len(b)",
    "        except Exception as e:",
    "            print(h['file_name'], 'PRED FAILED:', repr(e)[:200])",
    "            b, s = [], []",
    "        boxes_out.append({'file': h['file_name'],",
    "                          'boxes': [bi + [float(si)] for bi, si in zip(b, s)]})",
    "        times.append(time.time() - t1)",
    "    (OUT / 'deepforest_boxes.json').write_text(json.dumps(boxes_out))",
    "    report['stages']['deepforest'] = {'ok': True,",
    "        'mean_s_per_frame': round(sum(times) / max(len(times), 1), 2)}",
    "    print('deepforest done, mean s/frame:', report['stages']['deepforest']['mean_s_per_frame'])",
    "save()",
])))

C.append(("code", "\n".join([
    "import numpy as np",
    "try:",
    "    from pycocotools.coco import COCO",
    "    from pycocotools.cocoeval import COCOeval",
    "    names = {h['file_name'] for h in hold}",
    "    gimgs = [i for i in coco['images'] if i['file_name'] in names]",
    "    remap = {i['id']: n for n, i in enumerate(gimgs)}",
    "    gt = {'images': [{'id': n, 'width': i['width'], 'height': i['height']} for n, i in enumerate(gimgs)],",
    "          'annotations': [], 'categories': [{'id': 1, 'name': 'crop_tree'}]}",
    "    aid = 0",
    "    for a in coco['annotations']:",
    "        if a['image_id'] in remap and not a['iscrowd']:",
    "            gt['annotations'].append({'id': aid, 'image_id': remap[a['image_id']],",
    "                'category_id': 1, 'bbox': a['bbox'], 'area': a['area'], 'iscrowd': 0})",
    "            aid += 1",
    "    open('/tmp/gt.json', 'w').write(json.dumps(gt))",
    "    cocoGt = COCO('/tmp/gt.json')",
    "    inv = {}",
    "    for i in gimgs:",
    "        inv[i['file_name']] = remap[i['id']]",
    "    dt = []",
    "    pred = {e['file']: e['boxes'] for e in json.loads((OUT / 'deepforest_boxes.json').read_text())}",
    "    for f, boxes in pred.items():",
    "        for x0, y0, x1, y1, s in boxes:",
    "            dt.append({'image_id': inv[f], 'category_id': 1,",
    "                       'bbox': [x0, y0, x1 - x0, y1 - y0], 'score': float(s)})",
    "    if dt:",
    "        ev = COCOeval(cocoGt, cocoGt.loadRes(dt), 'bbox')",
    "        ev.evaluate(); ev.accumulate(); ev.summarize()",
    "        report['stages']['deepforest']['box_map50'] = float(ev.stats[1])",
    "    errs = []",
    "    for h in hold:",
    "        iid = next(i['id'] for i in gimgs if i['file_name'] == h['file_name'])",
    "        gn = sum(1 for a in coco['annotations'] if a['image_id'] == iid and not a['iscrowd'])",
    "        errs.append(abs(len(pred.get(h['file_name'], [])) - gn))",
    "    report['stages']['deepforest']['count_MAE'] = round(float(np.mean(errs)), 2)",
    "    print('mAP50:', report['stages']['deepforest'].get('box_map50'),",
    "          '| MAE:', report['stages']['deepforest']['count_MAE'])",
    "except Exception as e:",
    "    import traceback; traceback.print_exc()",
    "    report['stages'].setdefault('deepforest', {})['metrics_error'] = repr(e)",
    "save()",
])))

C.append(("code", "\n".join([
    "# ---- Dropped: tcd needs detectron2 (no pip wheel here) + flagship SegFormer is research-only ----",
    "report['stages']['tcd'] = {'ok': False,",
    "    'error': 'dropped: tcd-pipeline requires detectron2 (no matching distribution on this image); flagship SegFormer models are research-only licensed'}",
    "print('tcd dropped by design; Fase 2 vehicle = YOLO-seg fine-tune')",
    "save()",
    "print(json.dumps(report, indent=1))",
])))

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {
          "kernelspec": {"display_name": "Python 3", "language": "python",
                         "name": "python3"},
          "language_info": {"name": "python", "version": "3.10.0"}},
      "cells": []}
import uuid
for kind, src in C:
    cid = uuid.uuid4().hex[:8]
    if kind == "markdown":
        nb["cells"].append({"cell_type": "markdown", "id": cid, "metadata": {},
                            "source": src.splitlines(True)})
    else:
        nb["cells"].append({"cell_type": "code", "id": cid, "metadata": {},
                            "source": (src if src.endswith("\n") else src + "\n").splitlines(True),
                            "outputs": [], "execution_count": None})

with open(os.path.join(KDIR, "baseline-fase1.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("v2 cells:", len(nb["cells"]))
