"""Build kernels/fase4-t1/ — YOLO11n-seg fine-tune on gold_v2 (no pseudo).

T1 of the YOLO+SAM plan. Reads private dataset marcosalvarez234/limon-gold-v2.
Excludes the frozen 20-frame holdout from train/val with assertions.
Two farm-wise splits A/B, 150 epochs, imgsz 640, no aug cocktail.
Partial report always written to /kaggle/working/fase4_t1_report.json.

No CVAT, no uploads from this script. Local generation only.
"""

import json
import os
import uuid

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/fase4-t1"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# T1 — YOLO11n-seg limpio sobre gold_v2\n"
          "Dataset privado `limon-gold-v2` (405 frames, 4602 instancias, "
          "todo revisado, cero pseudo). Holdout 20 excluido de train y val "
          "con asserts. Splits por finca A/B. Reporte parcial siempre."))

C.append(("code", "\n".join([
    "import subprocess, sys",
    "for pkg in ['ultralytics', 'pycocotools']:",
    "    try: __import__(pkg); print(pkg, 'present')",
    "    except ImportError:",
    "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)",
    "        print(pkg, 'installed')",
    "import torch, os, glob, shutil",
    "print('cuda:', torch.cuda.is_available())",
    "assert torch.cuda.is_available(), 'Settings -> Accelerator -> GPU T4'",
    "print(torch.cuda.get_device_name(0), torch.__version__)",
    "# Diagnostico de montaje",
    "print('INPUT ARBOL:', [p for p in glob.glob('/kaggle/input/*')][:20])",
    "print('INPUT DATASETS:', [p for p in glob.glob('/kaggle/input/datasets/*/*')][:20])",
    "# Mount: Kaggle mounts datasets under /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug>",
    "roots = glob.glob('/kaggle/input/datasets/*/*') + glob.glob('/kaggle/input/*')",
    "cands = [p for p in roots",
    "         if os.path.exists(os.path.join(p, 'coco.json'))",
    "         and os.path.exists(os.path.join(p, 'splits.json'))",
    "         and os.path.exists(os.path.join(p, 'holdout.json'))]",
    "assert len(cands) >= 1, f'gold_v2 dataset no encontrado en {roots}'",
    "BASE = cands[0]",
    "print('BASE =', BASE, '| contenido:', os.listdir(BASE)[:12])",
    "# Kaggle input is READ-ONLY. Stage everything into /kaggle/working.",
    "DATA = os.path.join(BASE, 'all') if os.path.exists(os.path.join(BASE,'all','coco.json')) else BASE",
    "GOLD = '/kaggle/working/gold_v2'",
    "os.makedirs(os.path.join(GOLD, 'images'), exist_ok=True)",
    "os.makedirs(os.path.join(GOLD, 'yolo', 'labels'), exist_ok=True)",
    "for fn in ('coco.json','splits.json','holdout.json','quarantine.json','dev.json'):",
    "    src = os.path.join(DATA, fn)",
    "    if not os.path.exists(src): src = os.path.join(BASE, fn)",
    "    shutil.copyfile(src, os.path.join(GOLD, fn))",
    "# images + labels from extracted all.zip (read-only input -> writable staging)",
    "import glob as _g",
    "img_dirs = [os.path.join(DATA,'images'), os.path.join(BASE,'all','images'),",
    "            os.path.join(BASE,'images')]",
    "for d in img_dirs:",
    "    for p in _g.glob(os.path.join(d, '*.png')):",
    "        dst = os.path.join(GOLD, 'images', os.path.basename(p))",
    "        if not os.path.exists(dst): shutil.copyfile(p, dst)",
    "lab_dirs = [os.path.join(DATA,'yolo','labels'), os.path.join(BASE,'all','yolo','labels'),",
    "            os.path.join(BASE,'yolo','labels')]",
    "for d in lab_dirs:",
    "    for p in _g.glob(os.path.join(d, '*.txt')):",
    "        dst = os.path.join(GOLD, 'yolo', 'labels', os.path.basename(p))",
    "        if not os.path.exists(dst): shutil.copyfile(p, dst)",
    "print('BASE =', BASE, '| DATA =', DATA, '| GOLD =', GOLD,",
    "      '| imgs:', len(os.listdir(os.path.join(GOLD,'images'))),",
    "      '| labels:', len(os.listdir(os.path.join(GOLD,'yolo','labels'))))",
])))

C.append(("code", "\n".join([
    "import json, os, shutil, yaml",
    "from pathlib import Path",
    "OUT = Path('/kaggle/working')",
    "report = {'dataset': GOLD, 'splits': {}, 'holdout': {}, 'holdout_excluded': True}",
    "def save(): (OUT / 'fase4_t1_report.json').write_text(json.dumps(report, indent=1))",
    "# holdout.json can be {'holdout': [names]} (v2) or [dict] (v1)",
    "hraw = json.loads(Path(GOLD, 'holdout.json').read_text())",
    "if isinstance(hraw, dict):",
    "    hold_names = set(hraw['holdout'])",
    "else:",
    "    hold_names = {h['file_name'] for h in hraw}",
    "print('holdout frames:', len(hold_names))",
    "coco = json.loads(Path(GOLD, 'coco.json').read_text())",
    "byid = {i['id']: i['file_name'] for i in coco['images']}",
    "splits = json.loads(Path(GOLD, 'splits.json').read_text())",
    "ft = OUT / 'ft'; (ft / 'images').mkdir(parents=True, exist_ok=True)",
    "lab_src = Path(GOLD, 'yolo', 'labels')",
    "lab_dst = ft / 'labels'",
    "n = 0",
    "for i in coco['images']:",
    "    src = os.path.join(GOLD, 'images', i['file_name'])",
    "    dst = ft / 'images' / i['file_name']",
    "    if not os.path.exists(dst):",
    "        try: os.symlink(src, dst)",
    "        except Exception: shutil.copyfile(src, dst)",
    "    n += 1",
    "shutil.copytree(lab_src, lab_dst, dirs_exist_ok=True)",
    "print('staged images:', n, 'labels:', len(os.listdir(lab_dst)))",
    "assert n == len(coco['images']), 'staging incompleto'",
    "for name in ('A', 'B'):",
    "    sp = splits[f'split_{name}']",
    "    tr = [byid[i] for i in sp['train'] if byid[i] not in hold_names]",
    "    va = [byid[i] for i in sp['val'] if byid[i] not in hold_names]",
    "    assert not (set(tr) & set(va)), 'train/val overlap'",
    "    assert not (set(tr) | set(va)) & hold_names, 'HOLDOUT LEAK'",
    "    for fn in tr + va:",
    "        assert os.path.exists(ft/'images'/fn), f'falta imagen {fn}'",
    "    (OUT/f'train_{name}.txt').write_text('\\n'.join(str(ft/'images'/f) for f in tr)+'\\n')",
    "    (OUT/f'val_{name}.txt').write_text('\\n'.join(str(ft/'images'/f) for f in va)+'\\n')",
    "    (OUT/f'data_{name}.yaml').write_text(yaml.safe_dump(",
    "        {'path': str(ft), 'train': str(OUT/f'train_{name}.txt'),",
    "         'val': str(OUT/f'val_{name}.txt'), 'names': {0: 'crop_tree'}}))",
    "    report['splits'][name] = {'train': len(tr), 'val': len(va),",
    "        'holdout_removed': len(sp['train'])+len(sp['val'])-len(tr)-len(va)}",
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
    "                    project=str(OUT/'runs-t1'), name=f'split{name}',",
    "                    exist_ok=False, verbose=False)",
    "        # evaluate the SAVED best.pt, not the in-memory model",
    "        bw = OUT/'runs-t1'/f'split{name}'/'weights'/'best.pt'",
    "        model = YOLO(str(bw))",
    "        vm = model.val(data=str(OUT / f'data_{name}.yaml'), verbose=False)",
    "        report['splits'][name].update({'ok': True,",
    "            'box_map50': round(float(vm.box.map50),4),",
    "            'box_map5095': round(float(vm.box.map),4),",
    "            'mask_map50': round(float(vm.seg.map50),4),",
    "            'mask_map5095': round(float(vm.seg.map),4),",
    "            'weights': f'runs-t1/split{name}/weights/best.pt'})",
    "    except Exception as e:",
    "        import traceback; traceback.print_exc()",
    "        report['splits'][name].update({'ok': False, 'error': repr(e)[:400]})",
    "    save()",
])))

C.append(("code", "\n".join([
    "# Holdout 20 (real excluded) — splitA weights",
    "import numpy as np",
    "from pycocotools.coco import COCO",
    "from pycocotools.cocoeval import COCOeval",
    "from pycocotools import mask as cmask",
    "import torch.nn.functional as F",
    "try:",
    "    w = OUT/'runs-t1'/'splitA'/'weights'/'best.pt'",
    "    model = YOLO(str(w) if w.exists() else 'yolo11n-seg.pt')",
    "    paths = [os.path.join(GOLD,'images',h) for h in sorted(hold_names)",
    "             if os.path.exists(os.path.join(GOLD,'images',h))]",
    "    res = model.predict(paths, imgsz=640, conf=0.5, verbose=False)",
    "    gimgs = [i for i in coco['images'] if i['file_name'] in set(hold_names)]",
    "    remap = {i['id']: n for n, i in enumerate(gimgs)}",
    "    inv = {i['file_name']: remap[i['id']] for i in gimgs}",
    "    gt = {'images':[{'id':n,'width':512,'height':512} for n in range(len(gimgs))],",
    "          'annotations':[],'categories':[{'id':1,'name':'crop_tree'}]}",
    "    aid = 0",
    "    for a in coco['annotations']:",
    "        if a['image_id'] in remap and not a['iscrowd']:",
    "            gt['annotations'].append({'id':aid,'image_id':remap[a['image_id']],",
    "                'category_id':1,'bbox':a['bbox'],'segmentation':a['segmentation'],",
    "                'area':a['area'],'iscrowd':0}); aid += 1",
    "    open('/tmp/gt_t1.json','w').write(json.dumps(gt))",
    "    cocoGt = COCO('/tmp/gt_t1.json')",
    "    dt_box, dt_seg, errs = [], [], []",
    "    for r, p in zip(res, paths):",
    "        fn = os.path.basename(p); iid = inv[fn]; n = 0",
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
    "        gn = sum(1 for a in coco['annotations']",
    "              if a['image_id']==next(i['id'] for i in gimgs if i['file_name']==fn)",
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
    cell = {"cell_type": "markdown" if kind == "markdown" else "code",
            "id": cid, "metadata": {},
            "source": (src if src.endswith("\n") else src + "\n").splitlines(True)}
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    nb["cells"].append(cell)

with open(os.path.join(KDIR, "fase4-t1.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
meta = {"id": "marcosalvarez234/limon-t1-yolo-clean-on-gold-v2",
        "title": "Limon T1 - YOLO clean on gold_v2",
        "code_file": "fase4-t1.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": ["marcosalvarez234/limon-gold-v2d"],
        "competition_sources": [], "kernel_sources": []}
with open(os.path.join(KDIR, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("T1 kernel built:", len(nb["cells"]), "cells")
