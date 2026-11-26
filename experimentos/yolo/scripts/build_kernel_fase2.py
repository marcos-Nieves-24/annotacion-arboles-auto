"""Builds kernels/fase2-yolo/ (notebook + metadata) for Kaggle push.

Phase 2: fine-tune yolo11n-seg on gold_v1 with farm-wise splits
(A: train Villas 200 / val Magdala 131, B inverted), then holdout eval
(box+mask mAP vs non-iscrowd GT, count MAE). Defensive: partial report
always saved. No CVAT access.
"""

import json
import os
import uuid

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/fase2-yolo"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# Fase 2 — YOLO-seg fine-tune en oro 331 (limón)\n"
          "Splits por finca A/B + evaluación en holdout 20. "
          "Reporte parcial siempre en `/kaggle/working/fase2_report.json`."))

C.append(("code", "\n".join([
    "import torch, os, glob",
    "print('cuda:', torch.cuda.is_available())",
    "assert torch.cuda.is_available(), 'Settings -> Accelerator -> GPU T4'",
    "print(torch.cuda.get_device_name(0), torch.__version__)",
    "DS = None",
    "for root, dirs, files in os.walk('/kaggle/input'):",
    "    if 'holdout.json' in files:",
    "        DS = root; break",
    "    dirs[:] = [d for d in dirs if root.count(os.sep) - '/kaggle/input'.count(os.sep) < 4]",
    "assert DS is not None, 'gold dataset not mounted'",
    "print('DS =', DS)",
])))

C.append(("code", "\n".join([
    "import zipfile, shutil",
    "imgdir = os.path.join(DS, 'images')",
    "try:",
    "    for z in sorted(__import__('glob').glob(os.path.join(DS, '*.zip'))):",
    "        print('unzipping', os.path.basename(z), '...')",
    "        with zipfile.ZipFile(z) as zh:",
    "            zh.extractall(DS)",
    "    print('images:', len(os.listdir(imgdir)) if os.path.isdir(imgdir) else 'MISSING')",
    "    lab, want = os.path.join(DS, 'yolo', 'labels'), os.path.join(DS, 'labels')",
    "    if os.path.isdir(lab) and not os.path.exists(want):",
    "        try: os.symlink(lab, want)",
    "        except Exception: shutil.copytree(lab, want)",
    "        print('labels linked:', len(os.listdir(want)))",
    "except Exception as e:",
    "    print('UNZIP FAILED:', repr(e))",
])))

C.append(("code", "\n".join([
    "import importlib, subprocess",
    "for pkg, spec in [('ultralytics', 'ultralytics'), ('pycocotools', 'pycocotools')]:",
    "    try:",
    "        importlib.import_module(pkg); print(pkg, 'present')",
    "    except ImportError:",
    "        print('installing', spec)",
    "        subprocess.run(['pip', 'install', '-q', spec])",
    "import torch",
    "print('torch still cuda:', torch.cuda.is_available(), torch.__version__)",
])))

C.append(("code", "\n".join([
    "import json",
    "from pathlib import Path",
    "OUT = Path('/kaggle/working')",
    "report = {'splits': {}, 'holdout': {}}",
    "def save():",
    "    (OUT / 'fase2_report.json').write_text(json.dumps(report, indent=1))",
    "hold = json.loads(Path(DS, 'holdout.json').read_text())",
    "coco = json.loads(Path(DS, 'coco.json').read_text())",
    "splits = json.loads(Path(DS, 'splits.json').read_text())",
    "by_id = {i['id']: i['file_name'] for i in coco['images']}",
    "# stage writable tree in /kaggle/working (input mount is read-only):",
    "ft = OUT / 'ft'; (ft / 'images').mkdir(parents=True, exist_ok=True)",
    "import shutil",
    "n_link = n_copy = 0",
    "for i in coco['images']:",
    "    src = os.path.join(imgdir, i['file_name']); dst = ft / 'images' / i['file_name']",
    "    if not os.path.exists(dst):",
    "        try: os.symlink(src, dst); n_link += 1",
    "        except Exception: shutil.copyfile(src, dst); n_copy += 1",
    "shutil.copytree(os.path.join(DS, 'yolo', 'labels'), ft / 'labels', dirs_exist_ok=True)",
    "print(f'staged: {n_link} symlinks, {n_copy} copies, labels:', len(os.listdir(ft / 'labels')))",
    "# data.yaml per split with absolute image paths",
    "import yaml",
    "for name, sp in [('A', splits['split_A']), ('B', splits['split_B'])]:",
    "    tr = OUT / f'train_{name}.txt'; va = OUT / f'val_{name}.txt'",
    "    tr.write_text('\\n'.join(str(ft / 'images' / by_id[i]) for i in sp['train']) + '\\n')",
    "    va.write_text('\\n'.join(str(ft / 'images' / by_id[i]) for i in sp['val']) + '\\n')",
    "    (OUT / f'data_{name}.yaml').write_text(yaml.safe_dump(",
    "        {'path': str(ft), 'train': str(tr), 'val': str(va), 'names': {0: 'crop_tree'}}))",
    "    print(name, 'train:', len(sp['train']), 'val:', len(sp['val']))",
    "probe = ft / 'images' / by_id[splits['split_A']['train'][0]]",
    "plab = ft / 'labels' / (by_id[splits['split_A']['train'][0]].rsplit('.', 1)[0] + '.txt')",
    "assert probe.exists() and plab.exists(), f'stage broken: {probe.exists()} {plab.exists()}'",
    "print('stage verified:', probe.name, plab.stat().st_size, 'bytes')",
    "save()",
])))

C.append(("code", "\n".join([
    "from ultralytics import YOLO",
    "EPOCHS, IMGSZ = 150, 640",
    "RUN_SPLIT_B = True",
    "for name in (['A', 'B'] if RUN_SPLIT_B else ['A']):",
    "    try:",
    "        model = YOLO('yolo11n-seg.pt')",
    "        model.train(data=str(OUT / f'data_{name}.yaml'), epochs=EPOCHS, imgsz=IMGSZ,",
    "                    batch=-1, device=0, workers=2, patience=30,",
    "                    project=str(OUT / 'runs'), name=f'split{name}', exist_ok=True, verbose=False)",
    "        vm = model.val(verbose=False)",
    "        report['splits'][name] = {'ok': True,",
    "            'box_map50': round(float(vm.box.map50), 4),",
    "            'box_map5095': round(float(vm.box.map), 4),",
    "            'mask_map50': round(float(vm.seg.map50), 4),",
    "            'mask_map5095': round(float(vm.seg.map), 4),",
    "            'weights': f'runs/split{name}/weights/best.pt'}",
    "        print(name, report['splits'][name])",
    "    except Exception as e:",
    "        import traceback; traceback.print_exc()",
    "        report['splits'][name] = {'ok': False, 'error': repr(e)[:500]}",
    "        print(name, 'TRAIN FAILED')",
    "    save()",
])))

C.append(("code", "\n".join([
    "# ---- Holdout eval (non-iscrowd GT): box+mask mAP, count MAE ----",
    "import numpy as np",
    "from pycocotools.coco import COCO",
    "from pycocotools.cocoeval import COCOeval",
    "from pycocotools import mask as cmask",
    "from ultralytics import YOLO",
    "import torch.nn.functional as F",
    "try:",
    "    w = OUT / 'runs' / 'splitA' / 'weights' / 'best.pt'",
    "    model = YOLO(str(w) if w.exists() else 'yolo11n-seg.pt')",
    "    paths = [os.path.join(imgdir, h['file_name']) for h in hold]",
    "    res = model.predict(paths, imgsz=640, conf=0.25, verbose=False)",
    "    names = {h['file_name'] for h in hold}",
    "    gimgs = [i for i in coco['images'] if i['file_name'] in names]",
    "    remap = {i['id']: n for n, i in enumerate(gimgs)}",
    "    gt = {'images': [{'id': n, 'width': 512, 'height': 512} for n in range(len(gimgs))],",
    "          'annotations': [], 'categories': [{'id': 1, 'name': 'crop_tree'}]}",
    "    aid = 0",
    "    for a in coco['annotations']:",
    "        if a['image_id'] in remap and not a['iscrowd']:",
    "            gt['annotations'].append({'id': aid, 'image_id': remap[a['image_id']],",
    "                'category_id': 1, 'bbox': a['bbox'], 'segmentation': a['segmentation'],",
    "                'area': a['area'], 'iscrowd': 0})",
    "            aid += 1",
    "    open('/tmp/gt2.json', 'w').write(json.dumps(gt))",
    "    cocoGt = COCO('/tmp/gt2.json')",
    "    inv = {i['file_name']: remap[i['id']] for i in gimgs}",
    "    dt_box, dt_seg, errs = [], [], []",
    "    for r, h in zip(res, hold):",
    "        iid = inv[h['file_name']]",
    "        n = 0",
    "        if r.boxes is not None and len(r.boxes):",
    "            for xyxy, cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):",
    "                x0, y0, x1, y1 = xyxy",
    "                dt_box.append({'image_id': iid, 'category_id': 1,",
    "                    'bbox': [x0, y0, x1 - x0, y1 - y0], 'score': float(cf)})",
    "                n += 1",
    "        if r.masks is not None and len(r.masks):",
    "            import torch",
    "            md = r.masks.data.float()",
    "            if md.shape[1] != 512 or md.shape[2] != 512:",
    "                md = F.interpolate(md.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)",
    "            for k in range(md.shape[0]):",
    "                m = (md[k] > 0.5).cpu().numpy().astype('uint8')",
    "                rle = cmask.encode(np.asfortranarray(m))",
    "                if isinstance(rle, list): rle = rle[0]",
    "                c = rle['counts']",
    "                rle['counts'] = c.decode() if isinstance(c, bytes) else c",
    "                dt_seg.append({'image_id': iid, 'category_id': 1,",
    "                    'segmentation': rle, 'score': float(r.boxes.conf[k]) if r.boxes is not None and len(r.boxes) > k else 0.25})",
    "        gn = sum(1 for a in coco['annotations'] if a['image_id'] == next(i['id'] for i in gimgs if i['file_name'] == h['file_name']) and not a['iscrowd'])",
    "        errs.append(abs(n - gn))",
    "    out = {}",
    "    if dt_box:",
    "        ev = COCOeval(cocoGt, cocoGt.loadRes(dt_box), 'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()",
    "        out['box_map50'] = round(float(ev.stats[1]), 4); out['box_map5095'] = round(float(ev.stats[0]), 4)",
    "    if dt_seg:",
    "        ev = COCOeval(cocoGt, cocoGt.loadRes(dt_seg), 'segm'); ev.evaluate(); ev.accumulate(); ev.summarize()",
    "        out['mask_map50'] = round(float(ev.stats[1]), 4); out['mask_map5095'] = round(float(ev.stats[0]), 4)",
    "    out['count_MAE'] = round(float(np.mean(errs)), 2)",
    "    report['holdout'] = {'ok': True, **out}",
    "    print(out)",
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

with open(os.path.join(KDIR, "fase2-yolo.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)

meta = {
    "id": "marcosalvarez234/limon-fase2-yolo",
    "title": "Limon Fase2 Yolo",
    "code_file": "fase2-yolo.ipynb",
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
print("fase2 kernel built:", len(nb["cells"]), "cells")
