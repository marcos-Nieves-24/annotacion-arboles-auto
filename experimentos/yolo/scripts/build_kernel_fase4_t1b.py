"""Build kernels/fase4-t1b/ — SINGLE YOLO11n-seg on ALL gold_v2 frames.

Replaces the broken split A/B weights. One model trained on all non-holdout,
non-dev frames; validated on dev. No early stopping (patience=0) so best.pt
cannot be a collapsed epoch. Canonical weight copied to model_final.pt for
reliable download by T2 via kernel_source.

Dataset: marcosalvarez234/limon-gold-v2d
Output: /kaggle/working/model_final.pt + fase4_t1b_report.json
"""

import json
import os
import uuid

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/fase4-t1b"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# T1b — Modelo YOLO único sobre todo el oro\n"
          "Un solo YOLO11n-seg entrenado con 405 frames (holdout 20 y dev 38 excluidos). "
          "Sin early stopping. Peso canónico en `model_final.pt`. GPU T4."))

C.append(("code", "\n".join([
    "import subprocess, sys",
    "for pkg in ['ultralytics','pycocotools']:",
    "    try: __import__(pkg); print(pkg,'present')",
    "    except ImportError:",
    "        subprocess.run([sys.executable,'-m','pip','install','-q',pkg],check=True)",
    "import torch, os, glob, shutil, json, yaml",
    "print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))",
    "assert torch.cuda.is_available(), 'GPU T4 requerida'",
    "roots = glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/datasets/*/*')",
    "ds = [p for p in roots if os.path.exists(os.path.join(p,'coco.json'))",
    "      and os.path.exists(os.path.join(p,'splits.json'))]",
    "assert len(ds)==1, f'dataset no encontrado: {ds}'",
    "BASE = ds[0]",
    "DATA = os.path.join(BASE,'all') if os.path.exists(os.path.join(BASE,'all','images')) else BASE",
    "print('BASE =', BASE, '| DATA =', DATA)",
])))

C.append(("code", "\n".join([
    "from pathlib import Path",
    "OUT = Path('/kaggle/working')",
    "report = {'dataset': BASE, 'train': {}, 'dev': {}, 'holdout_excluded': True}",
    "def save(): (OUT/'fase4_t1b_report.json').write_text(json.dumps(report,indent=1))",
    "def jload(fn):",
    "    for d in (DATA, BASE):",
    "        p=os.path.join(d,fn)",
    "        if os.path.exists(p): return json.loads(Path(p).read_text())",
    "    raise FileNotFoundError(fn)",
    "coco = jload('coco.json')",
    "byid = {i['id']:i['file_name'] for i in coco['images']}",
    "hraw = jload('holdout.json')",
    "hold = set(hraw['holdout'] if isinstance(hraw,dict) else [h['file_name'] for h in hraw])",
    "dev = jload('dev.json')",
    "dev_ids = set(dev.get('A',[])+dev.get('B',[]))",
    "dev_names = {byid[i] for i in dev_ids if i in byid}",
    "all_names = {i['file_name'] for i in coco['images']}",
    "train_names = sorted(all_names - hold - dev_names)",
    "print('total:',len(all_names),'holdout:',len(hold),'dev:',len(dev_names),'train:',len(train_names))",
    "assert not (set(train_names) & hold), 'HOLDOUT LEAK'",
    "assert not (set(train_names) & dev_names), 'DEV LEAK'",
    "# stage writable",
    "ft = OUT/'ft'; (ft/'images').mkdir(parents=True,exist_ok=True)",
    "(ft/'labels').mkdir(parents=True,exist_ok=True)",
    "img_dirs=[os.path.join(DATA,'images'),os.path.join(BASE,'all','images'),os.path.join(BASE,'images')]",
    "for fn in all_names:",
    "    src=next((os.path.join(d,fn) for d in img_dirs if os.path.exists(os.path.join(d,fn))),None)",
    "    assert src, f'falta imagen {fn}'",
    "    dst=ft/'images'/fn",
    "    if not dst.exists(): shutil.copyfile(src,dst)",
    "lab_dirs=[os.path.join(DATA,'yolo','labels'),os.path.join(BASE,'all','yolo','labels'),os.path.join(BASE,'yolo','labels')]",
    "nlab=0",
    "for d in lab_dirs:",
    "    for p in glob.glob(os.path.join(d,'*.txt')):",
    "        dst=ft/'labels'/os.path.basename(p)",
    "        if not dst.exists(): shutil.copyfile(p,dst); nlab+=1",
    "print('staged:',len(os.listdir(ft/'images')),'imgs',nlab,'labels')",
    "(OUT/'train.txt').write_text('\\n'.join(str(ft/'images'/f) for f in train_names)+'\\n')",
    "(OUT/'val.txt').write_text('\\n'.join(str(ft/'images'/f) for f in sorted(dev_names))+'\\n')",
    "(OUT/'data.yaml').write_text(yaml.safe_dump(",
    "   {'path':str(ft),'train':str(OUT/'train.txt'),'val':str(OUT/'val.txt'),",
    "    'names':{0:'crop_tree'}}))",
    "report['train']={'n':len(train_names),'dev_n':len(dev_names),'labels':nlab}",
    "save()",
])))

C.append(("code", "\n".join([
    "from ultralytics import YOLO",
    "EPOCHS=120",
    "try:",
    "    model = YOLO('yolo11n-seg.pt')",
    "    model.train(data=str(OUT/'data.yaml'), epochs=EPOCHS, imgsz=640,",
    "                batch=-1, device=0, workers=2, patience=0,",
    "                project=str(OUT/'runs'), name='single', exist_ok=False, verbose=False)",
    "    bw = OUT/'runs'/'single'/'weights'/'best.pt'",
    "    assert bw.exists(), 'best.pt no generado'",
    "    shutil.copyfile(bw, OUT/'model_final.pt')",
    "    # eval on dev with the saved weight",
    "    m2 = YOLO(str(OUT/'model_final.pt'))",
    "    vm = m2.val(data=str(OUT/'data.yaml'), verbose=False)",
    "    report['dev'] = {'ok':True,",
    "        'box_map50':round(float(vm.box.map50),4),",
    "        'mask_map50':round(float(vm.seg.map50),4),",
    "        'mask_map5095':round(float(vm.seg.map),4)}",
    "    print('DEV mAP:', report['dev'])",
    "    # sanity check: count detections on a few frames",
    "    import cv2, numpy as np",
    "    tot=0; n=0",
    "    for fn in sorted(dev_names)[:10]:",
    "        p=ft/'images'/fn",
    "        if not p.exists(): continue",
    "        im=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)",
    "        r=m2.predict(im,imgsz=640,conf=0.25,verbose=False)[0]",
    "        tot += len(r.boxes) if r.boxes is not None else 0; n+=1",
    "    report['sanity']={'frames':n,'detections_conf025':tot}",
    "    print('SANITY:', report['sanity'])",
    "    assert tot > n*2, f'modelo colapsado: {tot} detecciones en {n} frames'",
    "except Exception as e:",
    "    import traceback; traceback.print_exc()",
    "    report['dev']={'ok':False,'error':repr(e)[:400]}",
    "save()",
])))

C.append(("code", "\n".join([
    "# Holdout eval once, with model_final.pt",
    "import numpy as np",
    "from pycocotools.coco import COCO",
    "from pycocotools.cocoeval import COCOeval",
    "from pycocotools import mask as cmask",
    "import torch.nn.functional as F",
    "try:",
    "    model = YOLO(str(OUT/'model_final.pt'))",
    "    hnames = sorted(h for h in hold if (ft/'images'/h).exists())",
    "    paths = [str(ft/'images'/h) for h in hnames]",
    "    res = model.predict(paths, imgsz=640, conf=0.5, verbose=False)",
    "    gimgs = [i for i in coco['images'] if i['file_name'] in set(hnames)]",
    "    remap = {i['id']: n for n,i in enumerate(gimgs)}",
    "    inv = {i['file_name']: remap[i['id']] for i in gimgs}",
    "    gt = {'images':[{'id':n,'width':512,'height':512} for n in range(len(gimgs))],",
    "          'annotations':[],'categories':[{'id':1,'name':'crop_tree'}]}",
    "    aid=0",
    "    for a in coco['annotations']:",
    "        if a['image_id'] in remap and not a['iscrowd']:",
    "            gt['annotations'].append({'id':aid,'image_id':remap[a['image_id']],",
    "                'category_id':1,'bbox':a['bbox'],'segmentation':a['segmentation'],",
    "                'area':a['area'],'iscrowd':0}); aid+=1",
    "    open('/tmp/gt1b.json','w').write(json.dumps(gt))",
    "    cocoGt=COCO('/tmp/gt1b.json')",
    "    dt_box,dt_seg,errs=[],[],[]",
    "    for r,h in zip(res,hnames):",
    "        iid=inv[h]; n=0",
    "        if r.boxes is not None and len(r.boxes):",
    "            for xyxy,cf in zip(r.boxes.xyxy.tolist(),r.boxes.conf.tolist()):",
    "                x0,y0,x1,y1=xyxy",
    "                dt_box.append({'image_id':iid,'category_id':1,",
    "                    'bbox':[x0,y0,x1-x0,y1-y0],'score':float(cf)}); n+=1",
    "        if r.masks is not None and len(r.masks):",
    "            md=r.masks.data.float()",
    "            if md.shape[1]!=512 or md.shape[2]!=512:",
    "                md=F.interpolate(md.unsqueeze(0),size=(512,512),mode='nearest').squeeze(0)",
    "            for k in range(md.shape[0]):",
    "                m=(md[k]>0.5).cpu().numpy().astype('uint8')",
    "                rle=cmask.encode(np.asfortranarray(m))",
    "                if isinstance(rle,list): rle=rle[0]",
    "                c=rle['counts']; rle['counts']=c.decode() if isinstance(c,bytes) else c",
    "                cf=r.boxes.conf[k].item() if r.boxes is not None and len(r.boxes)>k else 0.5",
    "                dt_seg.append({'image_id':iid,'category_id':1,'segmentation':rle,'score':float(cf)})",
    "        gn=sum(1 for a in coco['annotations']",
    "              if a['image_id']==next(i['id'] for i in gimgs if i['file_name']==h)",
    "              and not a['iscrowd'])",
    "        errs.append(abs(n-gn))",
    "    out={}",
    "    if dt_box:",
    "        ev=COCOeval(cocoGt,cocoGt.loadRes(dt_box),'bbox'); ev.evaluate(); ev.accumulate()",
    "        out['box_map50']=round(float(ev.stats[1]),4)",
    "    if dt_seg:",
    "        ev=COCOeval(cocoGt,cocoGt.loadRes(dt_seg),'segm'); ev.evaluate(); ev.accumulate()",
    "        out['mask_map50']=round(float(ev.stats[1]),4)",
    "    out['count_MAE']=round(float(np.mean(errs)),2)",
    "    report['holdout']={'ok':True, **out}; print('HOLDOUT:',out)",
    "except Exception as e:",
    "    import traceback; traceback.print_exc()",
    "    report['holdout']={'ok':False,'error':repr(e)[:400]}",
    "save()",
    "print(json.dumps(report,indent=1))",
])))

nb = {"nbformat":4,"nbformat_minor":5,"metadata":{
    "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.10.0"}},"cells":[]}
for kind,src in C:
    cid=uuid.uuid4().hex[:8]
    cell={"cell_type":"markdown" if kind=="markdown" else "code","id":cid,"metadata":{},
          "source":(src if src.endswith("\n") else src+"\n").splitlines(True)}
    if kind=="code": cell["outputs"]=[]; cell["execution_count"]=None
    nb["cells"].append(cell)
with open(os.path.join(KDIR,"fase4-t1b.ipynb"),"w") as f: json.dump(nb,f,indent=1)
meta={"id":"marcosalvarez234/limon-t1b-single-model",
      "title":"Limon T1b Single YOLO All Gold",
      "code_file":"fase4-t1b.ipynb","language":"python","kernel_type":"notebook",
      "is_private":"true","enable_gpu":"true","enable_internet":"true",
      "dataset_sources":["marcosalvarez234/limon-gold-v2d"],
      "competition_sources":[],"kernel_sources":[]}
with open(os.path.join(KDIR,"kernel-metadata.json"),"w") as f: json.dump(meta,f,indent=1)
print("T1b kernel built:",len(nb["cells"]),"cells")
