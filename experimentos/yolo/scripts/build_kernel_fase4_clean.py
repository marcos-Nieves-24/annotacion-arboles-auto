"""Build kernels/fase4-clean/ — YOLO11n-seg fine-tune on GOLD ONLY.

Differences vs ronda1b:
  - NO pseudo-labels. Gold 331 images (2912 non-iscrowd instances).
  - Holdout 20 explicitly excluded from BOTH train and val.
  - Two farm-wise splits (A: train Villas / val Magdala, B inverted).
  - Same hypers as Fase 2 (150 ep, imgsz 640, no aug cocktail).
  - Partial report always written to /kaggle/working/fase4_clean_report.json.

No CVAT, no uploads. Local generation only.
"""

import json
import os
import uuid

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/fase4-clean"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# Fase 4 — Reentrenamiento limpio (solo oro 331)\n"
          "YOLO11n-seg, splits por finca A/B, holdout 20 excluido de train y val. "
          "Cero pseudo-labels. Reporte parcial siempre en "
          "`/kaggle/working/fase4_clean_report.json`."))

C.append(("code", "\n".join([
    "import torch, os",
    "print('cuda:', torch.cuda.is_available())",
    "assert torch.cuda.is_available(), 'Settings -> Accelerator -> GPU T4'",
    "print(torch.cuda.get_device_name(0), torch.__version__)",
    "GOLD = next(d for d in [p for p in __import__('glob').glob('/kaggle/input/datasets/*/*')]",
    "           if os.path.exists(os.path.join(d, 'holdout.json')))",
    "print('GOLD =', GOLD)",
])))

C.append(("code", "\n".join([
    "import zipfile, shutil",
    "for z in sorted(__import__('glob').glob(os.path.join(GOLD, '*.zip'))):",
    "    print('unzip', os.path.basename(z))",
    "    with zipfile.ZipFile(z) as zh: zh.extractall(GOLD)",
    "lab = os.path.join(GOLD, 'yolo', 'labels')",
    "want = os.path.join(GOLD, 'labels')",
    "if os.path.isdir(lab) and not os.path.exists(want):",
    "    try: os.symlink(lab, want)",
    "    except Exception: shutil.copytree(lab, want)",
    "print('images:', len(os.listdir(os.path.join(GOLD, 'images'))))",
])))

C.append(("code", "\n".join([
    "import json, os, shutil",
    "from pathlib import Path",
    "from collections import defaultdict",
    "OUT = Path('/kaggle/working')",
    "report = {'splits': {}, 'holdout': {}, 'holdout_excluded': True}",
    "def save(): (OUT / 'fase4_clean_report.json').write_text(json.dumps(report, indent=1))",
    "hold = json.loads(Path(GOLD, 'holdout.json').read_text())",
    "hold_names = {h['file_name'] for h in hold}",
    "coco = json.loads(Path(GOLD, 'coco.json').read_text())",
    "byid = {i['id']: i['file_name'] for i in coco['images']}",
    "splits = json.loads(Path(GOLD, 'splits.json').read_text())",
    "# stage writable tree",
    "ft = OUT / 'ft'; (ft / 'images').mkdir(parents=True, exist_ok=True)",
    "for i in coco['images']:",
    "    src = os.path.join(GOLD, 'images', i['file_name'])",
    "    dst = ft / 'images' / i['file_name']",
    "    if not os.path.exists(dst):",
    "        try: os.symlink(src, dst)",
    "        except Exception: shutil.copyfile(src, dst)",
    "shutil.copytree(os.path.join(GOLD, 'yolo', 'labels'), ft / 'labels', dirs_exist_ok=True)",
    "import yaml",
    "for name in ('A', 'B'):",
    "    sp = splits[f'split_{name}']",
    "    tr = [byid[i] for i in sp['train'] if byid[i] not in hold_names]",
    "    va = [byid[i] for i in sp['val'] if byid[i] not in hold_names]",
    "    assert not (set(tr) & set(va)), 'train/val overlap'",
    "    assert not (set(tr) | set(va)) & hold_names, 'holdout leaked into split'",
    "    (OUT / f'train_{name}.txt').write_text('\\n'.join(str(ft/'images'/f) for f in tr)+'\\n')",
    "    (OUT / f'val_{name}.txt').write_text('\\n'.join(str(ft/'images'/f) for f in va)+'\\n')",
    "    (OUT / f'data_{name}.yaml').write_text(yaml.safe_dump(",
    "        {'path': str(ft), 'train': str(OUT/f'train_{name}.txt'),",
    "         'val': str(OUT/f'val_{name}.txt'), 'names': {0: 'crop_tree'}}))",
    "    report['splits'][name] = {'train': len(tr), 'val': len(va),",
    "                              'holdout_removed': len(sp['train'])+len(sp['val'])-len(tr)-len(va)}",
    "    print(name, report['splits'][name])",
    "save()",
])))

C.append(("code", "\n".join([
    "from ultralytics import YOLO",
    "EPOCHS, IMGSZ = 150, 640",
    "for name in ('A', 'B'):",
    "    try:",
    "        model = YOLO('yolo11n-seg.pt')",
    "        model.train(data=str(OUT / f'data_{name}.yaml'), epochs=EPOCHS, imgsz=IMGSZ,",
    "                    batch=-1, device=0, workers=2, patience=30,",
    "                    project=str(OUT/'runs-clean'), name=f'split{name}',",
    "                    exist_ok=True, verbose=False)",
    "        vm = model.val(verbose=False)",
    "        report['splits'][name].update({'ok': True,",
    "            'box_map50': round(float(vm.box.map50),4),",
    "            'box_map5095': round(float(vm.box.map),4),",
    "            'mask_map50': round(float(vm.seg.map50),4),",
    "            'mask_map5095': round(float(vm.seg.map),4),",
    "            'weights': f'runs-clean/split{name}/weights/best.pt'})",
    "    except Exception as e:",
    "        import traceback; traceback.print_exc()",
    "        report['splits'][name].update({'ok': False, 'error': repr(e)[:400]})",
    "    save()",
])))

C.append(("code", "\n".join([
    "# Holdout eval (20 frames, excluded from train) — splitA weights",
    "import numpy as np",
    "from pycocotools.coco import COCO",
    "from pycocotools.cocoeval import COCOeval",
    "from pycocotools import mask as cmask",
    "from ultralytics import YOLO",
    "import torch.nn.functional as F",
    "try:",
    "    w = OUT/'runs-clean'/'splitA'/'weights'/'best.pt'",
    "    model = YOLO(str(w) if w.exists() else 'yolo11n-seg.pt')",
    "    paths = [os.path.join(GOLD,'images',h['file_name']) for h in hold]",
    "    res = model.predict(paths, imgsz=640, conf=0.5, verbose=False)",
    "    names = {h['file_name'] for h in hold}",
    "    gimgs = [i for i in coco['images'] if i['file_name'] in names]",
    "    remap = {i['id']: n for n, i in enumerate(gimgs)}",
    "    gt = {'images':[{'id':n,'width':512,'height':512} for n in range(len(gimgs))],",
    "          'annotations':[],'categories':[{'id':1,'name':'crop_tree'}]}",
    "    aid = 0",
    "    for a in coco['annotations']:",
    "        if a['image_id'] in remap and not a['iscrowd']:",
    "            gt['annotations'].append({'id':aid,'image_id':remap[a['image_id']],",
    "                'category_id':1,'bbox':a['bbox'],'segmentation':a['segmentation'],",
    "                'area':a['area'],'iscrowd':0}); aid += 1",
    "    open('/tmp/gt4.json','w').write(json.dumps(gt))",
    "    cocoGt = COCO('/tmp/gt4.json')",
    "    inv = {i['file_name']: remap[i['id']] for i in gimgs}",
    "    dt_box, dt_seg, errs = [], [], []",
    "    for r, h in zip(res, hold):",
    "        iid = inv[h['file_name']]; n = 0",
    "        if r.boxes is not None and len(r.boxes):",
    "            for xyxy, cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):",
    "                x0,y0,x1,y1 = xyxy",
    "                dt_box.append({'image_id':iid,'category_id':1,",
    "                    'bbox':[x0,y0,x1-x0,y1-y0],'score':float(cf)}); n += 1",
    "        if r.masks is not None and len(r.masks):",
    "            md = r.masks.data.float()",
    "            if md.shape[1]!=512 or md.shape[2]!=512:",
    "                md = F.interpolate(md.unsqueeze(0),size=(512,512),mode='nearest').squeeze(0)",
    "            for k in range(md.shape[0]):",
    "                m=(md[k]>0.5).cpu().numpy().astype('uint8')",
    "                rle=cmask.encode(np.asfortranarray(m))",
    "                if isinstance(rle,list): rle=rle[0]",
    "                c=rle['counts']; rle['counts']=c.decode() if isinstance(c,bytes) else c",
    "                cf=r.boxes.conf[k].item() if r.boxes is not None and len(r.boxes)>k else 0.5",
    "                dt_seg.append({'image_id':iid,'category_id':1,",
    "                    'segmentation':rle,'score':float(cf)})",
    "        gn=sum(1 for a in coco['annotations']",
    "              if a['image_id']==next(i['id'] for i in gimgs if i['file_name']==h['file_name'])",
    "              and not a['iscrowd'])",
    "        errs.append(abs(n-gn))",
    "    out = {}",
    "    if dt_box:",
    "        ev=COCOeval(cocoGt,cocoGt.loadRes(dt_box),'bbox'); ev.evaluate(); ev.accumulate()",
    "        out['box_map50']=round(float(ev.stats[1]),4); out['box_map5095']=round(float(ev.stats[0]),4)",
    "    if dt_seg:",
    "        ev=COCOeval(cocoGt,cocoGt.loadRes(dt_seg),'segm'); ev.evaluate(); ev.accumulate()",
    "        out['mask_map50']=round(float(ev.stats[1]),4); out['mask_map5095']=round(float(ev.stats[0]),4)",
    "    out['count_MAE']=round(float(np.mean(errs)),2)",
    "    report['holdout']={'ok':True, **out}; print(out)",
    "except Exception as e:",
    "    import traceback; traceback.print_exc()",
    "    report['holdout']={'ok':False,'error':repr(e)[:400]}",
    "save()",
    "print(json.dumps(report, indent=1))",
])))

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}}, "cells": []}
for kind, src in C:
    cid = uuid.uuid4().hex[:8]
    nb["cells"].append({"cell_type": "markdown" if kind == "markdown" else "code",
                        "id": cid, "metadata": {},
                        "source": (src if src.endswith("\n") else src + "\n").splitlines(True),
                        **({} if kind == "markdown" else {"outputs": [], "execution_count": None})})

with open(os.path.join(KDIR, "fase4-clean.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
meta = {"id": "marcosalvarez234/limon-fase4-clean",
        "title": "Limon Fase4 Clean Gold-Only",
        "code_file": "fase4-clean.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": ["marcosalvarez234/limon-gold-v1"],
        "competition_sources": [], "kernel_sources": []}
with open(os.path.join(KDIR, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("fase4-clean kernel built:", len(nb["cells"]), "cells")
