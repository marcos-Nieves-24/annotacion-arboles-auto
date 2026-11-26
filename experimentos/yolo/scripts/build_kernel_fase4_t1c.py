"""Build kernels/fase4-t1c/ — retrain with patience=30 + hard frames x2.

Fixes the likely cause of T1b's poor fit (41% of its own train undetected):
  - patience=30 (not 0) -> early stopping picks the real best epoch
  - reload best.pt before val (no in-memory model)
  - 137 hard frames (>=50% crowns missed) repeated x2 in train.txt
  - epochs 150, exist_ok=False, same base hypers

Dataset: marcosalvarez234/limon-gold-v2d. GPU T4. No CVAT writes.
"""

import json, os, uuid

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/fase4-t1c"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# T1c — Reentrenamiento corregido\n"
          "Fix de T1b: patience=30, recarga de best.pt antes de validar, "
          "137 frames duros x2. Dataset gold_v2d. GPU T4."))

C.append(("code", "\n".join([
    "import subprocess, sys",
    "for pkg in ['ultralytics','pycocotools']:",
    "    try: __import__(pkg); print(pkg,'present')",
    "    except ImportError:",
    "        subprocess.run([sys.executable,'-m','pip','install','-q',pkg],check=True)",
    "import torch, os, glob, shutil, json, yaml, random",
    "print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))",
    "assert torch.cuda.is_available(), 'GPU T4'",
    "roots = glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/datasets/*/*')",
    "ds = [p for p in roots if os.path.exists(os.path.join(p,'coco.json'))",
    "      and os.path.exists(os.path.join(p,'splits.json'))]",
    "assert ds, f'dataset no encontrado: {roots}'",
    "BASE = ds[0]",
    "DATA = os.path.join(BASE,'all') if os.path.exists(os.path.join(BASE,'all','images')) else BASE",
    "print('BASE =', BASE, '| DATA =', DATA)",
])))

C.append(("code", "\n".join([
    "from pathlib import Path",
    "OUT = Path('/kaggle/working')",
    "report = {'dataset': BASE, 'splits': {}, 'holdout': {}}",
    "def save(): (OUT/'fase4_t1c_report.json').write_text(json.dumps(report,indent=1))",
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
    "dev_names = {byid[i] for i in dev.get('A',[])+dev.get('B',[]) if i in byid}",
    "all_names = {i['file_name'] for i in coco['images']}",
    "train_names = sorted(all_names - hold - dev_names)",
    "# hard frames: those with >=50% gold crowns missed by T1b (hard-coded list from local analysis)",
    "HARD = set(['Limon_LasVillas_00178.png','Limon_LasVillas_00250.png','Limon_LasVillas_00299.png','Limon_LasVillas_00300.png','Limon_LasVillas_00301.png','Limon_LasVillas_00303.png','Limon_LasVillas_00352.png','Limon_LasVillas_00354.png','Limon_LasVillas_00356.png','Limon_LasVillas_00357.png','Limon_LasVillas_00388.png','Limon_LasVillas_00389.png','Limon_LasVillas_00394.png','Limon_LasVillas_00404.png','Limon_LasVillas_00406.png','Limon_LasVillas_00409.png','Limon_LasVillas_00411.png','Limon_LasVillas_00441.png','Limon_LasVillas_00447.png','Limon_LasVillas_00451.png','Limon_LasVillas_00452.png','Limon_LasVillas_00456.png','Limon_LasVillas_00458.png','Limon_LasVillas_00459.png','Limon_LasVillas_00461.png','Limon_LasVillas_00462.png','Limon_LasVillas_00463.png','Limon_LasVillas_00504.png','Limon_LasVillas_00514.png','Limon_LasVillas_00517.png','Limon_LasVillas_00518.png','Limon_LasVillas_00545.png','Limon_LasVillas_00560.png','Limon_LasVillas_00561.png','Limon_LasVillas_00562.png','Limon_LasVillas_00570.png','Limon_LasVillas_00610.png','Limon_LasVillas_00614.png','Limon_LasVillas_00615.png','Limon_LasVillas_00616.png','Limon_LasVillas_00617.png','Limon_LasVillas_00618.png','Limon_LasVillas_00619.png','Limon_LasVillas_00620.png','Limon_LasVillas_00626.png','Limon_LasVillas_00662.png','Limon_LasVillas_00665.png','Limon_LasVillas_00666.png','Limon_LasVillas_00667.png','Limon_LasVillas_00668.png','Limon_LasVillas_00669.png','Limon_LasVillas_00670.png','Limon_Magdala_00383.png','Limon_Magdala_00384.png','Limon_Magdala_00463.png','Limon_Magdala_00464.png','Limon_Magdala_00545.png','Limon_Magdala_00546.png','Limon_Magdala_00625.png','Limon_Magdala_00627.png','Limon_Magdala_00786.png','Limon_Magdala_00804.png','Limon_Magdala_00805.png','Limon_Magdala_00862.png','Limon_Magdala_00863.png','Limon_Magdala_00867.png','Limon_Magdala_00868.png','Limon_Magdala_00869.png','Limon_Magdala_00870.png','Limon_Magdala_00884.png','Limon_Magdala_00885.png','Limon_Magdala_00886.png','Limon_Magdala_00925.png','Limon_Magdala_00926.png','Limon_Magdala_00927.png','Limon_Magdala_00928.png','Limon_Magdala_00929.png','Limon_Magdala_00931.png','Limon_Magdala_00943.png','Limon_Magdala_00944.png','Limon_Magdala_00945.png','Limon_Magdala_00947.png','Limon_Magdala_00948.png','Limon_Magdala_00949.png','Limon_Magdala_00950.png','Limon_Magdala_00951.png','Limon_Magdala_00952.png','Limon_Magdala_00953.png','Limon_Magdala_00965.png','Limon_Magdala_00966.png','Limon_Magdala_00967.png','Limon_Magdala_00968.png','Limon_Magdala_00970.png','Limon_Magdala_01006.png','Limon_Magdala_01007.png','Limon_Magdala_01008.png','Limon_Magdala_01009.png','Limon_Magdala_01023.png','Limon_Magdala_01024.png','Limon_Magdala_01025.png','Limon_Magdala_01026.png','Limon_Magdala_01028.png','Limon_Magdala_01029.png','Limon_Magdala_01030.png','Limon_Magdala_01031.png','Limon_Magdala_01032.png','Limon_Magdala_01033.png','Limon_Magdala_01034.png','Limon_Magdala_01048.png','Limon_Magdala_01050.png','Limon_Magdala_01051.png','Limon_Magdala_01052.png','Limon_Magdala_01053.png','Limon_Magdala_01086.png','Limon_Magdala_01087.png','Limon_Magdala_01088.png','Limon_LasVillas_00671.png','Limon_LasVillas_00679.png','Limon_LasVillas_00680.png','Limon_LasVillas_00714.png','Limon_LasVillas_00718.png','Limon_LasVillas_00719.png','Limon_LasVillas_00720.png','Limon_LasVillas_00721.png','Limon_Magdala_01191.png','Limon_Magdala_01192.png','Limon_Magdala_01193.png','Limon_Magdala_01194.png','Limon_Magdala_01195.png','Limon_Magdala_01212.png','Limon_Magdala_01247.png','Limon_Magdala_01248.png','Limon_Magdala_01251.png','Limon_Magdala_01263.png','Limon_Magdala_01265.png','Limon_Magdala_01266.png','Limon_Magdala_01267.png'])",
    "print('hard frames in kernel:', len(HARD))",
    "assert not (HARD & hold) and not (HARD & dev_names), 'hard frames contaminan eval'",
    "# stage images+labels",
    "ft = OUT/'ft'; (ft/'images').mkdir(parents=True,exist_ok=True); (ft/'labels').mkdir(parents=True,exist_ok=True)",
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
    "# train.txt with hard frames x2",
    "train_list = list(train_names) + [f for f in train_names if f in HARD]",
    "(OUT/'train.txt').write_text('\\n'.join(str(ft/'images'/f) for f in train_list)+'\\n')",
    "(OUT/'val.txt').write_text('\\n'.join(str(ft/'images'/f) for f in sorted(dev_names))+'\\n')",
    "(OUT/'data.yaml').write_text(yaml.safe_dump({'path':str(ft),",
    "   'train':str(OUT/'train.txt'),'val':str(OUT/'val.txt'),'names':{0:'crop_tree'}}))",
    "report['train']={'n':len(train_names),'hard_x2':len([f for f in train_names if f in HARD]),",
    "                 'dev':len(dev_names),'holdout':len(hold),'labels':nlab}",
    "print(report['train']); save()",
])))

C.append(("code", "\n".join([
    "from ultralytics import YOLO",
    "EPOCHS=150",
    "try:",
    "    model = YOLO('yolo11n-seg.pt')",
    "    model.train(data=str(OUT/'data.yaml'), epochs=EPOCHS, imgsz=640,",
    "                batch=-1, device=0, workers=2, patience=30,",
    "                project=str(OUT/'runs'), name='t1c', exist_ok=False, verbose=False)",
    "    bw = OUT/'runs'/'t1c'/'weights'/'best.pt'",
    "    assert bw.exists(), 'best.pt no generado'",
    "    shutil.copyfile(bw, OUT/'model_final.pt')",
    "    model = YOLO(str(OUT/'model_final.pt'))  # reload saved best",
    "    vm = model.val(data=str(OUT/'data.yaml'), verbose=False)",
    "    report['dev'] = {'ok':True,",
    "        'box_map50':round(float(vm.box.map50),4),",
    "        'mask_map50':round(float(vm.seg.map50),4),",
    "        'mask_map5095':round(float(vm.seg.map),4)}",
    "    print('DEV:',report['dev'])",
    "    # sanity: detections on 10 dev frames",
    "    import cv2, numpy as np",
    "    tot=0;n=0",
    "    for fn in sorted(dev_names)[:10]:",
    "        p=ft/'images'/fn",
    "        if not p.exists(): continue",
    "        im=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)",
    "        r=model.predict(im,imgsz=640,conf=0.25,verbose=False)[0]",
    "        tot+=len(r.boxes) if r.boxes is not None else 0; n+=1",
    "    report['sanity']={'frames':n,'det_conf025':tot}",
    "    print('SANITY:',report['sanity'])",
    "    assert tot > n*2, f'modelo colapsado: {tot} det en {n} frames'",
    "except Exception as e:",
    "    import traceback; traceback.print_exc()",
    "    report['dev']={'ok':False,'error':repr(e)[:400]}",
    "save()",
])))

C.append(("code", "\n".join([
    "# Holdout eval once with final model",
    "import numpy as np",
    "from pycocotools.coco import COCO",
    "from pycocotools.cocoeval import COCOeval",
    "from pycocotools import mask as cmask",
    "import torch.nn.functional as F",
    "try:",
    "    model = YOLO(str(OUT/'model_final.pt'))",
    "    hnames = sorted(h for h in hold if (ft/'images'/h).exists())",
    "    res = model.predict([str(ft/'images'/h) for h in hnames], imgsz=640, conf=0.5, verbose=False)",
    "    gimgs = [i for i in coco['images'] if i['file_name'] in set(hnames)]",
    "    remap = {i['id']:n for n,i in enumerate(gimgs)}",
    "    inv = {i['file_name']:remap[i['id']] for i in gimgs}",
    "    gt = {'images':[{'id':n,'width':512,'height':512} for n in range(len(gimgs))],",
    "          'annotations':[],'categories':[{'id':1,'name':'crop_tree'}]}",
    "    aid=0",
    "    for a in coco['annotations']:",
    "        if a['image_id'] in remap and not a['iscrowd']:",
    "            gt['annotations'].append({'id':aid,'image_id':remap[a['image_id']],",
    "                'category_id':1,'bbox':a['bbox'],'segmentation':a['segmentation'],",
    "                'area':a['area'],'iscrowd':0}); aid+=1",
    "    open('/tmp/gt1c.json','w').write(json.dumps(gt))",
    "    cocoGt = COCO('/tmp/gt1c.json')",
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
    "                dt_seg.append({'image_id':iid,'category_id':1,",
    "                    'segmentation':rle,'score':float(cf)})",
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
with open(os.path.join(KDIR,"fase4-t1c.ipynb"),"w") as f: json.dump(nb,f,indent=1)
meta={"id":"marcosalvarez234/limon-fase4-t1c-retrain",
      "title":"Limon Fase4 T1c Retrain",
      "code_file":"fase4-t1c.ipynb","language":"python","kernel_type":"notebook",
      "is_private":"true","enable_gpu":"true","enable_internet":"true",
      "dataset_sources":["marcosalvarez234/limon-gold-v2d"],
      "competition_sources":[],"kernel_sources":[]}
with open(os.path.join(KDIR,"kernel-metadata.json"),"w") as f: json.dump(meta,f,indent=1)
print("T1c kernel built:", len(nb["cells"]), "cells")
