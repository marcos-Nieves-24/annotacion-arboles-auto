"""Build kernels/fase4-t2/ — YOLO(T1 weights) + SAM box-prompt pipeline.

T2 of the YOLO+SAM plan. Runs on Kaggle T4 (SAM needs GPU).
  - kernel_source: limon-t1-yolo-clean-on-gold-v2  (best.pt from T1)
  - dataset_source: limon-gold-v2b                 (images + coco + splits + dev)

Pipeline: YOLO boxes (conf 0.5, NMS 0.35) -> SAM ViT-B box prompt -> masks
          -> center-in-box filter -> GSD filter -> COCO/YOLO-seg + overlays
Eval on dev split: YOLO-only vs YOLO+SAM (P/R/F1, over/under-split, count MAE).
Holdout 20 touched once at the end. Partial report always saved.

No CVAT. Local generation only.
"""

import json
import os
import uuid

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/fase4-t2"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# T2 — YOLO + SAM (CASCADE + multimask)\n"
          "Pesos YOLO de T1 + SAM ViT-B con box de cada copa como prompt. "
          "Filtro centro-en-box + GSD. Eval en dev (YOLO solo vs YOLO+SAM). "
          "Holdout 20 al final. GPU T4 requerida."))

C.append(("code", "\n".join([
    "import subprocess, sys",
    "for pkg in ['ultralytics', 'pycocotools', 'segment-anything', 'opencv-python']:",
    "    try: __import__(pkg.replace('-','_')); print(pkg, 'present')",
    "    except ImportError:",
    "        subprocess.run([sys.executable,'-m','pip','install','-q',pkg], check=True)",
    "        print(pkg, 'installed')",
    "import torch, os, glob, shutil, json",
    "print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))",
    "assert torch.cuda.is_available(), 'GPU T4 requerida para SAM'",
    "# --- locate SINGLE YOLO weight from T1b kernel source ---",
    "roots = glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/datasets/*/*')",
    "print('INPUTS:', roots[:20])",
    "W = glob.glob('/kaggle/input/**/model_final.pt', recursive=True)",
    "assert W, f'peso unico T1b no encontrado en {roots}'",
    "YOLO_W = W[0]",
    "print('YOLO_W =', YOLO_W)",
    "# --- locate gold_v2b dataset ---",
    "ds = [p for p in roots if os.path.exists(os.path.join(p,'coco.json'))]",
    "assert ds, 'dataset gold_v2b no encontrado'",
    "BASE = ds[0]",
    "DATA = os.path.join(BASE,'all') if os.path.exists(os.path.join(BASE,'all','images')) else BASE",
    "print('BASE =', BASE, '| DATA =', DATA)",
])))

C.append(("code", "\n".join([
    "import urllib.request",
    "SAM_CKPT = '/kaggle/working/sam_vit_b_01ec64.pth'",
    "if not os.path.exists(SAM_CKPT):",
    "    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'",
    "    print('descargando SAM ViT-B...')",
    "    urllib.request.urlretrieve(url, SAM_CKPT)",
    "print('SAM ckpt:', os.path.getsize(SAM_CKPT), 'bytes')",
    "from segment_anything import sam_model_registry, SamPredictor",
    "sam = sam_model_registry['vit_b'](checkpoint=SAM_CKPT).to('cuda')",
    "predictor = SamPredictor(sam)",
    "from ultralytics import YOLO",
    "ymodel = YOLO(YOLO_W)  # single model trained on all gold",
    "print('modelo listo (unico, todo el oro)')",
])))

C.append(("code", "\n".join([
    "import numpy as np, cv2, yaml",
    "from pathlib import Path",
    "OUT = Path('/kaggle/working')",
    "report = {'yolo_weights': YOLO_W, 'sam': 'vit_b', 'dev': {}, 'holdout': {}}",
    "def save(): (OUT/'fase4_t2_report.json').write_text(json.dumps(report, indent=1))",
    "def find_json(fn):",
    "    for d in (DATA, BASE):",
    "        p = os.path.join(d, fn)",
    "        if os.path.exists(p): return p",
    "    raise FileNotFoundError(fn)",
    "coco = json.loads(Path(find_json('coco.json')).read_text())",
    "byid = {i['id']: i['file_name'] for i in coco['images']}",
    "farm_of = {i['file_name']: i.get('farm','villas' if 'Villas' in i['file_name'] else 'magdala')",
    "           for i in coco['images']}",
    "AREAS = {'villas':(882,39224), 'magdala':(410,18240)}",
    "hraw = json.loads(Path(find_json('holdout.json')).read_text())",
    "hold = set(hraw['holdout'] if isinstance(hraw,dict) else [h['file_name'] for h in hraw])",
    "dev = json.loads(Path(find_json('dev.json')).read_text())",
    "dev_names = [byid[i] for i in dev.get('A',[])+dev.get('B',[]) if i in byid]",
    "eval_names = [n for n in dev_names if n not in hold]",
    "print('dev frames:', len(eval_names))",
    "def area_mask(m): return int(m.sum())",
    "def center(m):",
    "    ys,xs=np.where(m); return (float(xs.mean()),float(ys.mean())) if len(xs) else (-1,-1)",
    "def in_box(c,b):",
    "    x0,y0,x1,y1=b; return x0<=c[0]<=x1 and y0<=c[1]<=y1",
])))

C.append(("code", "\n".join([
    "# --- inference on dev: YOLO-only vs YOLO+SAM ---",
    "tp_y=fp_y=fn_y=tp_s=fp_s=fn_s=0",
    "os.makedirs(OUT/'overlays', exist_ok=True)",
    "for name in eval_names:",
    "    p = os.path.join(DATA,'images',name)",
    "    if not os.path.exists(p): continue",
    "    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)",
    "    farm = farm_of[name]",
    "    res = ymodel.predict(img, imgsz=640, conf=0.5, iou=0.35, verbose=False)[0]",
    "    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))",
    "    # YOLO-only masks",
    "    yolo_masks = []",
    "    if res.masks is not None and len(res.masks):",
    "        md = res.masks.data.float().cpu().numpy()",
    "        for k in range(md.shape[0]):",
    "            m = md[k] > 0.5",
    "            if m.shape != (512,512):",
    "                m = cv2.resize(m.astype('uint8'),(512,512)) > 0.5",
    "            yolo_masks.append(m)",
    "    # SAM CASCADE refinement: box -> 3 masks -> best logits -> box+mask -> final",
    "    predictor.set_image(img)",
    "    sam_masks = []",
    "    for b in boxes:",
    "        x0,y0,x1,y1 = b",
    "        if x1<=x0 or y1<=y0: continue",
    "        # YOLO boxes are in 640-space (imgsz=640); SAM image is 512 -> rescale",
    "        sc = 512.0/640.0; PAD = 15",
    "        bx = np.array([max(0,x0*sc-PAD),max(0,y0*sc-PAD),",
    "                       min(512,x1*sc+PAD),min(512,y1*sc+PAD)],dtype=np.float32)",
    "        # step 1: box prompt, 3 masks, pick best by SAM's own score",
    "        m1, s1, l1 = predictor.predict(box=bx, multimask_output=True)",
    "        best = int(np.argmax(s1))",
    "        # step 2: cascade with the best logits as mask prompt",
    "        m2, s2, l2 = predictor.predict(box=bx, mask_input=l1[best:best+1],",
    "                                       multimask_output=False)",
    "        m = m2[0]",
    "        a = area_mask(m)",
    "        lo,hi = AREAS[farm]",
    "        if not (lo<=a<=hi): continue",
    "        if not in_box(center(m), tuple(bx)): continue",
    "        sam_masks.append(m)",
    "    # reference (non-iscrowd) masks from COCO",
    "    iid = next(i['id'] for i in coco['images'] if i['file_name']==name)",
    "    refs = []",
    "    for a in coco['annotations']:",
    "        if a['image_id']==iid and not a['iscrowd'] and a.get('segmentation'):",
    "            seg=a['segmentation'][0]; pts=np.array(seg,float).reshape(-1,2).astype(int)",
    "            m=np.zeros((512,512),np.uint8); cv2.fillPoly(m,[pts],1); refs.append(m.astype(bool))",
    "    def match(preds):",
    "        pairs=[]",
    "        for gi,g in enumerate(refs):",
    "            for pi,p in enumerate(preds):",
    "                inter=np.logical_and(g,p).sum(); union=np.logical_or(g,p).sum()",
    "                if union and inter/union>=0.5: pairs.append((inter/union,gi,pi))",
    "        pairs.sort(reverse=True); ug,up,tp=set(),set(),0",
    "        for _,gi,pi in pairs:",
    "            if gi not in ug and pi not in up: ug.add(gi);up.add(pi);tp+=1",
    "        return tp, len(preds)-tp, len(refs)-tp",
    "    a,b,c = match(yolo_masks); tp_y+=a;fp_y+=b;fn_y+=c",
    "    a,b,c = match(sam_masks);  tp_s+=a;fp_s+=b;fn_s+=c",
    "    # overlay SAM masks on image",
    "    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)",
    "    for m in sam_masks:",
    "        cnts,_=cv2.findContours(m.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)",
    "        cv2.drawContours(vis,cnts,-1,(0,0,255),2)",
    "    cv2.imwrite(str(OUT/'overlays'/name), vis)",
    "def metrics(tp,fp,fn):",
    "    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0",
    "    return {'P':round(P,4),'R':round(R,4),'F1':round(2*P*R/(P+R),4) if P+R else 0}",
    "report['dev']={'yolo_only':metrics(tp_y,fp_y,fn_y), 'yolo_sam':metrics(tp_s,fp_s,fn_s),",
    "               'yolo':{'tp':tp_y,'fp':fp_y,'fn':fn_y},",
    "               'yolo_sam_counts':{'tp':tp_s,'fp':fp_s,'fn':fn_s}}",
    "print('DEV:', report['dev'])",
    "save()",
])))

C.append(("code", "\n".join([
    "# Holdout 20 evaluated ONCE, with the winning branch from dev",
    "winner = 'yolo_sam' if report['dev']['yolo_sam']['F1'] >= report['dev']['yolo_only']['F1'] else 'yolo_only'",
    "report['winner'] = winner",
    "tp=fp=fn=0",
    "for name in sorted(hold):",
    "    p = os.path.join(DATA,'images',name)",
    "    if not os.path.exists(p): continue",
    "    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)",
    "    farm = farm_of[name]",
    "    res = ymodel.predict(img, imgsz=640, conf=0.5, iou=0.35, verbose=False)[0]",
    "    if winner=='yolo_only':",
    "        preds=[]",
    "        if res.masks is not None:",
    "            md=res.masks.data.float().cpu().numpy()",
    "            preds=[]",
    "            for k in range(md.shape[0]):",
    "                m=md[k]>0.5",
    "                if m.shape!=(512,512):",
    "                    m=cv2.resize(m.astype('uint8'),(512,512))>0.5",
    "                preds.append(m)",
    "    else:",
    "        predictor.set_image(img); preds=[]",
    "        boxes=res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))",
    "        for b in boxes:",
    "            sc=512.0/640.0; PAD=15",
    "            bx=np.array([max(0,b[0]*sc-PAD),max(0,b[1]*sc-PAD),",
    "                         min(512,b[2]*sc+PAD),min(512,b[3]*sc+PAD)],dtype='float32')",
    "            m1,s1,l1=predictor.predict(box=bx,multimask_output=True)",
    "            best=int(np.argmax(s1))",
    "            m2,_,_=predictor.predict(box=bx,mask_input=l1[best:best+1],multimask_output=False)",
    "            m=m2[0]; lo,hi=AREAS[farm]",
    "            if lo<=area_mask(m)<=hi and in_box(center(m),tuple(bx)): preds.append(m)",
    "    iid=next(i['id'] for i in coco['images'] if i['file_name']==name)",
    "    refs=[]",
    "    for a in coco['annotations']:",
    "        if a['image_id']==iid and not a['iscrowd'] and a.get('segmentation'):",
    "            pts=np.array(a['segmentation'][0],float).reshape(-1,2).astype(int)",
    "            m=np.zeros((512,512),np.uint8);cv2.fillPoly(m,[pts],1);refs.append(m.astype(bool))",
    "    pairs=[]",
    "    for gi,g in enumerate(refs):",
    "        for pi,pm in enumerate(preds):",
    "            it=np.logical_and(g,pm).sum();un=np.logical_or(g,pm).sum()",
    "            if un and it/un>=0.5: pairs.append((it/un,gi,pi))",
    "    pairs.sort(reverse=True);ug,up=set(),set();ltp=0",
    "    for _,gi,pi in pairs:",
    "        if gi not in ug and pi not in up: ug.add(gi);up.add(pi);ltp+=1",
    "    tp+=ltp; fp+=len(preds)-ltp; fn+=len(refs)-ltp",
    "report['holdout']={'branch':winner, **metrics(tp,fp,fn),",
    "                   'tp':tp,'fp':fp,'fn':fn}",
    "save()",
    "print('HOLDOUT:', report['holdout'])",
    "print(json.dumps(report,indent=1))",
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

with open(os.path.join(KDIR, "fase4-t2.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
meta = {"id": "marcosalvarez234/limon-t2-yolo-sam-box-prompt",
        "title": "Limon T2 - YOLO SAM box prompt",
        "code_file": "fase4-t2.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true",
        "kernel_sources": ["marcosalvarez234/limon-t1b-single-yolo-all-gold"],
        "dataset_sources": ["marcosalvarez234/limon-gold-v2d"],
        "competition_sources": []}
with open(os.path.join(KDIR, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("T2 kernel built:", len(nb["cells"]), "cells")
