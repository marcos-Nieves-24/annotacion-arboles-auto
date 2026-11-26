"""Build kernels/texture-classifier/ — ResNet-18 binary crown/background classifier.

Trains on the 640x640 patches from marcosalvarez234/limon-texture-patches
(copy-paste texture fill, Rossi 2026 protocol). GPU T4, 30 epochs, patience 5.
Outputs model.pth + report. No CVAT writes.
"""

import json, os, uuid

KDIR = "/home/biohacker24/annotacion-arboles-auto/kernels/texture-classifier"
os.makedirs(KDIR, exist_ok=True)

C = []

C.append(("markdown",
          "# Texture classifier v2 — ResNet-18 raw crops (no fill)\n"
          "Parches 640x640 con copy-paste de textura (Rossi 2026). "
          "Entrena binario con AdamW 1e-4, 30 ep, patience 5. GPU T4."))

C.append(("code", "\n".join([
    "import torch, torchvision, os, glob, json, random",
    "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())",
    "assert torch.cuda.is_available(), 'GPU T4 requerida'",
    "roots = glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/datasets/*/*')",
    "ds = [p for p in roots if os.path.isdir(os.path.join(p,'patches','pos','train'))]",
    "assert ds, f'dataset de parches no encontrado: {roots}'",
    "BASE = os.path.join(ds[0], 'patches')",
    "print('BASE =', BASE)",
    "for s in ('train','val'):",
    "    print(s, 'pos:', len(glob.glob(os.path.join(BASE,'pos',s,'*.jpg'))),",
    "          'soil:', len(glob.glob(os.path.join(BASE,'neg_soil',s,'*.jpg'))),",
    "          'veg:', len(glob.glob(os.path.join(BASE,'neg_veg',s,'*.jpg'))),",
    "          'partial:', len(glob.glob(os.path.join(BASE,'neg_partial',s,'*.jpg'))))",
])))

C.append(("code", "\n".join([
    "from torchvision import transforms, models",
    "from torch.utils.data import Dataset, DataLoader",
    "from PIL import Image",
    "import torch.nn as nn",
    "SEED=7; random.seed(SEED); torch.manual_seed(SEED)",
    "class Patches(Dataset):",
    "    def __init__(self, base, split):",
    "        self.items = []",
    "        for label, name in [(1,'pos'),(0,'neg_soil'),(0,'neg_veg'),(0,'neg_partial')]:",
    "            for p in glob.glob(os.path.join(base,name,split,'*.jpg')):",
    "                self.items.append((p,label))",
    "        random.Random(SEED).shuffle(self.items)",
    "        self.tf = transforms.Compose([",
    "            transforms.Resize((640,640)),",
    "            transforms.RandomHorizontalFlip(),",
    "            transforms.RandomVerticalFlip(),",
    "            transforms.RandomRotation(15),",
    "            transforms.ToTensor(),",
    "            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])",
    "    def __len__(self): return len(self.items)",
    "    def __getitem__(self,i):",
    "        p,label=self.items[i]",
    "        im=Image.open(p).convert('RGB')",
    "        return self.tf(im), label",
    "tr = DataLoader(Patches(BASE,'train'), batch_size=32, shuffle=True, num_workers=2)",
    "va = DataLoader(Patches(BASE,'val'), batch_size=32, shuffle=False, num_workers=2)",
    "print('batches train/val:', len(tr), len(va))",
    "model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)",
    "model.fc = nn.Linear(model.fc.in_features, 1)",
    "model = model.cuda()",
    "crit = nn.BCEWithLogitsLoss()",
    "opt = torch.optim.AdamW(model.parameters(), lr=1e-4)",
])))

C.append(("code", "\n".join([
    "import numpy as np, time",
    "EPOCHS=30; PATIENCE=5; best=1e9; wait=0; history=[]",
    "for ep in range(EPOCHS):",
    "    model.train(); tl=0; n=0",
    "    for x,y in tr:",
    "        x=x.cuda(); y=y.float().cuda()",
    "        opt.zero_grad(); out=model(x).squeeze(1)",
    "        loss=crit(out,y); loss.backward(); opt.step()",
    "        tl+=loss.item()*len(y); n+=len(y)",
    "    model.eval(); vl=0; tp=fp=fn=tn=0",
    "    with torch.no_grad():",
    "        for x,y in va:",
    "            x=x.cuda(); y=y.float().cuda()",
    "            out=model(x).squeeze(1); loss=crit(out,y)",
    "            vl+=loss.item()*len(y)",
    "            pred=(torch.sigmoid(out)>0.5).float()",
    "            tp+=int(((pred==1)&(y==1)).sum()); fp+=int(((pred==1)&(y==0)).sum())",
    "            fn+=int(((pred==0)&(y==1)).sum()); tn+=int(((pred==0)&(y==0)).sum())",
    "    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0",
    "    F1=2*P*R/(P+R) if P+R else 0; vloss=vl/len(va.dataset)",
    "    history.append({'ep':ep+1,'train_loss':round(tl/n,4),'val_loss':round(vloss,4),",
    "                    'P':round(P,4),'R':round(R,4),'F1':round(F1,4)})",
    "    print(f'ep {ep+1}: loss={tl/n:.4f}/{vloss:.4f} P={P:.3f} R={R:.3f} F1={F1:.3f}', flush=True)",
    "    if vloss < best - 1e-4:",
    "        best=vloss; wait=0; torch.save(model.state_dict(),'/kaggle/working/model.pth')",
    "    else:",
    "        wait+=1",
    "        if wait>=PATIENCE: print('early stop'); break",
    "print('best val loss:', best)",
])))

C.append(("code", "\n".join([
    "# final metrics with best weights",
    "model.load_state_dict(torch.load('/kaggle/working/model.pth'))",
    "model.eval(); tp=fp=fn=tn=0",
    "with torch.no_grad():",
    "    for x,y in va:",
    "        x=x.cuda(); y=y.float().cuda()",
    "        pred=(torch.sigmoid(model(x).squeeze(1))>0.5).float()",
    "        tp+=int(((pred==1)&(y==1)).sum()); fp+=int(((pred==1)&(y==0)).sum())",
    "        fn+=int(((pred==0)&(y==1)).sum()); tn+=int(((pred==0)&(y==0)).sum())",
    "P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0",
    "F1=2*P*R/(P+R) if P+R else 0",
    "report={'model':'resnet18','protocol':'Rossi2026 copy-paste',",
    "        'val':{'P':round(P,4),'R':round(R,4),'F1':round(F1,4),",
    "               'tp':tp,'fp':fp,'fn':fn,'tn':tn},",
    "        'history':history}",
    "json.dump(report, open('/kaggle/working/texture_report.json','w'), indent=1)",
    "print(json.dumps(report['val'], indent=1))",
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
with open(os.path.join(KDIR,"texture-classifier.ipynb"),"w") as f: json.dump(nb,f,indent=1)
meta={"id":"marcosalvarez234/limon-texture-classifier-v3-hardneg",
      "title":"Limon Texture Classifier V3 HardNeg",
      "code_file":"texture-classifier.ipynb","language":"python",
      "kernel_type":"notebook","is_private":"true","enable_gpu":"true",
      "enable_internet":"true",
      "dataset_sources":["marcosalvarez234/limon-texture-patches-v3"],
      "competition_sources":[],"kernel_sources":[]}
with open(os.path.join(KDIR,"kernel-metadata.json"),"w") as f: json.dump(meta,f,indent=1)
print("texture kernel built:", len(nb["cells"]), "cells")
