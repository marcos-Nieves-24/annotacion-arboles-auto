"""Builds kernels/ronda1b/ for Kaggle push.

Ronda 1b retrain: gold (x2 oversampled ~ weight) + R1 pseudo, val = pure
gold splitA-val. Baseline hypers (NO aug cocktail: P1 proved it hurts).
Gate: val mask must stay within 1pt of Fase2 A (83.1).
"""

import json
import os
import uuid

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/ronda1b"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# Ronda 1b — gold x2 + pseudo, val oro puro\n"
          "Compuerta: val mask >= 82.1 (Fase2 A 83.1 - 1pt). "
          "Hipers baseline, sin cocktail."))

C.append(("code", "\n".join([
    "import torch, os",
    "print('cuda:', torch.cuda.is_available())",
    "assert torch.cuda.is_available(), 'Settings -> Accelerator -> GPU T4'",
    "print(torch.cuda.get_device_name(0), torch.__version__)",
    "GOLD = next(d for d in [p for p in __import__('glob').glob('/kaggle/input/datasets/*/*')] if os.path.exists(os.path.join(d, 'holdout.json')))",
    "R1 = next(d for d in [p for p in __import__('glob').glob('/kaggle/input/datasets/*/*')] if os.path.exists(os.path.join(d, 'pseudo_report.json')))",
    "print('GOLD =', GOLD)",
    "print('R1 =', R1)",
])))

C.append(("code", "\n".join([
    "import zipfile, shutil",
    "for ds in (GOLD, R1):",
    "    for z in sorted(__import__('glob').glob(os.path.join(ds, '*.zip'))):",
    "        print('unzipping', os.path.basename(z), 'from', os.path.basename(ds))",
    "        with zipfile.ZipFile(z) as zh:",
    "            zh.extractall(ds)",
    "print('gold:', sorted(os.listdir(GOLD))[:8])",
    "print('r1:', sorted(os.listdir(R1))[:8])",
])))

C.append(("code", "\n".join([
    "import importlib, subprocess",
    "for pkg, spec in [('ultralytics', 'ultralytics'), ('pycocotools', 'pycocotools'), ('yaml', 'pyyaml')]:",
    "    try: importlib.import_module(pkg); print(pkg, 'present')",
    "    except ImportError:",
    "        print('installing', spec); subprocess.run(['pip', 'install', '-q', spec])",
    "import torch",
    "print('torch still cuda:', torch.cuda.is_available())",
])))

C.append(("code", "\n".join([
    "import json",
    "from pathlib import Path",
    "OUT = Path('/kaggle/working')",
    "report = {}",
    "def save():",
    "    (OUT / 'ronda1b_report.json').write_text(json.dumps(report, indent=1))",
    "# stage writable tree",
    "FT = OUT / 'ft2'; (FT / 'images').mkdir(parents=True, exist_ok=True); (FT / 'labels').mkdir(parents=True, exist_ok=True)",
    "def link_or_copy(src, dst):",
    "    import os as _os",
    "    if _os.path.exists(dst): return",
    "    try: _os.symlink(src, dst)",
    "    except Exception: shutil.copyfile(src, dst)",
    "# gold images+labels",
    "gi = os.path.join(GOLD, 'images'); gl = os.path.join(GOLD, 'yolo', 'labels')",
    "n = 0",
    "for f in sorted(os.listdir(gi)):",
    "    link_or_copy(os.path.join(gi, f), str(FT / 'images' / f)); n += 1",
    "for f in sorted(os.listdir(gl)):",
    "    shutil.copyfile(os.path.join(gl, f), str(FT / 'labels' / f))",
    "print('gold staged images:', n)",
    "# pseudo images+labels (prefix to avoid collisions)",
    "ri = os.path.join(R1, 'images'); rl = os.path.join(R1, 'labels')",
    "if not os.path.isdir(ri):  # zips may nest one level",
    "    import glob as _g",
    "    cands = [d for d in _g.glob(os.path.join(R1, '*')) if os.path.isdir(d)]",
    "    print('r1 subdirs:', [os.path.basename(d) for d in cands])",
    "np_ = 0",
    "for root, dirs, files in os.walk(R1):",
    "    for f in files:",
    "        if f.endswith('.png'): link_or_copy(os.path.join(root, f), str(FT / 'images' / ('ps_' + f))); np_ += 1",
    "        elif f.endswith('.txt') and 'labels' in root.replace(chr(92), '/'):",
    "            shutil.copyfile(os.path.join(root, f), str(FT / 'labels' / ('ps_' + f)))",
    "print('pseudo staged images:', np_)",
    "# train = gold splitA-train x2 + pseudo all; val = gold splitA-val pure",
    "splits = json.loads(Path(GOLD, 'splits.json').read_text())",
    "by_id = {i['id']: i['file_name'] for i in json.loads(Path(GOLD, 'coco.json').read_text())['images']}",
    "tr = [by_id[i] for i in splits['split_A']['train']] * 2",
    "tr += ['ps_' + f for f in sorted(os.listdir(FT / 'images')) if f.startswith('ps_')]",
    "va = [by_id[i] for i in splits['split_A']['val']]",
    "(OUT / 'train_r1.txt').write_text('\\n'.join(str(FT / 'images' / f) for f in tr) + '\\n')",
    "(OUT / 'val_r1.txt').write_text('\\n'.join(str(FT / 'images' / f) for f in va) + '\\n')",
    "import yaml",
    "(OUT / 'data_r1.yaml').write_text(yaml.safe_dump({'path': str(FT), 'train': str(OUT / 'train_r1.txt'), 'val': str(OUT / 'val_r1.txt'), 'names': {0: 'crop_tree'}}))",
    "print('train:', len(tr), 'val:', len(va))",
    "report['train_n'] = len(tr); report['val_n'] = len(va); save()",
])))

C.append(("code", "\n".join([
    "from ultralytics import YOLO",
    "try:",
    "    model = YOLO('yolo11n-seg.pt')",
    "    model.train(data=str(OUT / 'data_r1.yaml'), epochs=150, imgsz=640,",
    "                batch=-1, device=0, workers=2, patience=30,",
    "                project=str(OUT / 'runs1b'), name='r1', exist_ok=True, verbose=False)",
    "    vm = model.val(verbose=False)",
    "    report['val'] = {'ok': True, 'box_map50': round(float(vm.box.map50), 4),",
    "        'mask_map50': round(float(vm.seg.map50), 4), 'mask_map5095': round(float(vm.seg.map), 4)}",
    "    print('VAL:', report['val'])",
    "    print('GATE(>=82.1):', 'PASS' if report['val']['mask_map50'] >= 0.821 else 'FAIL')",
    "except Exception as e:",
    "    import traceback; traceback.print_exc()",
    "    report['val'] = {'ok': False, 'error': repr(e)[:500]}",
    "save()",
])))

C.append(("code", "\n".join([
    "# holdout with r1 weights (splitA best if r1 missing)",
    "import numpy as np",
    "from pycocotools.coco import COCO",
    "from pycocotools.cocoeval import COCOeval",
    "from pycocotools import mask as cmask",
    "from ultralytics import YOLO",
    "import torch.nn.functional as F",
    "from pathlib import Path as _P",
    "cands = list((OUT / 'runs1b' / 'r1' / 'weights').glob('best.pt')) if (OUT / 'runs1b').exists() else []",
    "w = str(cands[0]) if cands else 'yolo11n-seg.pt'",
    "print('weights:', w)",
    "try:",
    "    import glob as _g",
    "    gimg = _g.glob(os.path.join(GOLD, 'images', '*.png'))",
    "    hold = json.loads(_P(GOLD, 'holdout.json').read_text())",
    "    coco = json.loads(_P(GOLD, 'coco.json').read_text())",
    "    model = YOLO(w)",
    "    paths = [os.path.join(GOLD, 'images', h['file_name']) for h in hold]",
    "    res = model.predict(paths, imgsz=640, conf=0.5, verbose=False)",
    "    names = {h['file_name'] for h in hold}",
    "    gimgs = [i for i in coco['images'] if i['file_name'] in names]",
    "    remap = {i['id']: n for n, i in enumerate(gimgs)}",
    "    gt = {'images': [{'id': n, 'width': 512, 'height': 512} for n in range(len(gimgs))], 'annotations': [], 'categories': [{'id': 1, 'name': 'crop_tree'}]}",
    "    aid = 0",
    "    for a in coco['annotations']:",
    "        if a['image_id'] in remap and not a['iscrowd']:",
    "            gt['annotations'].append({'id': aid, 'image_id': remap[a['image_id']], 'category_id': 1, 'bbox': a['bbox'], 'segmentation': a['segmentation'], 'area': a['area'], 'iscrowd': 0}); aid += 1",
    "    open('/tmp/gt3.json', 'w').write(json.dumps(gt))",
    "    cocoGt = COCO('/tmp/gt3.json')",
    "    inv = {i['file_name']: remap[i['id']] for i in gimgs}",
    "    dt_box, dt_seg, errs = [], [], []",
    "    import torch",
    "    for r, h in zip(res, hold):",
    "        iid = inv[h['file_name']]; n = 0",
    "        confs = r.boxes.conf.tolist() if r.boxes is not None and len(r.boxes) else []",
    "        boxes = r.boxes.xyxy.tolist() if r.boxes is not None and len(r.boxes) else []",
    "        for (x0, y0, x1, y1), cf in zip(boxes, confs):",
    "            dt_box.append({'image_id': iid, 'category_id': 1, 'bbox': [x0, y0, x1 - x0, y1 - y0], 'score': float(cf)}); n += 1",
    "        if r.masks is not None and len(r.masks):",
    "            md = r.masks.data.float()",
    "            if md.shape[1] != 512 or md.shape[2] != 512: md = F.interpolate(md.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)",
    "            for k in range(md.shape[0]):",
    "                m = (md[k] > 0.5).cpu().numpy().astype('uint8')",
    "                rle = cmask.encode(np.asfortranarray(m))",
    "                c = rle['counts']; rle['counts'] = c.decode() if isinstance(c, bytes) else c",
    "                dt_seg.append({'image_id': iid, 'category_id': 1, 'segmentation': rle, 'score': float(confs[k]) if k < len(confs) else 0.25})",
    "        gn = sum(1 for a in coco['annotations'] if a['image_id'] == next(i['id'] for i in gimgs if i['file_name'] == h['file_name']) and not a['iscrowd'])",
    "        errs.append(abs(n - gn))",
    "    out = {}",
    "    if dt_box:",
    "        ev = COCOeval(cocoGt, cocoGt.loadRes(dt_box), 'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()",
    "        out['box_map50'] = round(float(ev.stats[1]), 4)",
    "    if dt_seg:",
    "        ev = COCOeval(cocoGt, cocoGt.loadRes(dt_seg), 'segm'); ev.evaluate(); ev.accumulate(); ev.summarize()",
    "        out['mask_map50'] = round(float(ev.stats[1]), 4); out['mask_map5095'] = round(float(ev.stats[0]), 4)",
    "    out['count_MAE'] = round(float(np.mean(errs)), 2)",
    "    report['holdout'] = {'ok': True, **out}; print(out)",
    "except Exception as e:",
    "    import traceback; traceback.print_exc()",
    "    report['holdout'] = {'ok': False, 'error': repr(e)[:500]}",
    "save()",
    "print(json.dumps(report, indent=1))",
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

with open(os.path.join(KDIR, "ronda1b.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)

meta = {
    "id": "marcosalvarez234/limon-ronda1b",
    "title": "Limon Ronda1b Retrain",
    "code_file": "ronda1b.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": ["marcosalvarez234/limon-gold-v1",
                        "marcosalvarez234/limon-r1-pseudo"],
    "competition_sources": [],
    "kernel_sources": []
}
with open(os.path.join(KDIR, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("ronda1b kernel built:", len(nb["cells"]), "cells")
