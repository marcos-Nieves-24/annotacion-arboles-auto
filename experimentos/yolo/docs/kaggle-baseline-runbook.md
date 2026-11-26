# Runbook — baseline Fase 1 en Kaggle (GPU T4, internet ON)

Dataset: `marcosalvarez234/limon-gold-v1` (privado). Attach al notebook.
Holdout fijo: `holdout.json` (20 frames, 10 Villas + 10 Magdala). No se entrena sobre él.

## Settings del notebook
- Accelerator: GPU T4. Internet: ON (baja pesos de HF). Environment: latest.
- Attach data: `marcosalvarez234/limon-gold-v1`.

## Celda 1 — GPU + rutas
```python
import torch, os, zipfile
assert torch.cuda.is_available(), "Activar GPU: Settings -> Accelerator -> GPU T4"
print(torch.cuda.get_device_name(0))
DS = "/kaggle/input/limon-gold-v1"
print(sorted(os.listdir(DS)))
# v2 trae images/ como zip: descomprimir una vez
z = os.path.join(DS, "images.zip")
if os.path.exists(z) and not os.path.isdir(os.path.join(DS, "images")):
    with zipfile.ZipFile(z) as zh:
        zh.extractall(DS)
print("images:", len(os.listdir(os.path.join(DS, "images"))))
```

## Celda 2 — instalar (sin tocar torch de la imagen)
```python
!pip install -q deepforest
!pip install -q git+https://github.com/restor-foundation/tcd.git
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
```
Si algo intenta bajar torch de CPU: frenar, fijar `torch==<versión de la imagen>` y reintentar.

## Celda 3 — holdout + GT
```python
import json
from pathlib import Path
hold = json.loads(Path(f"{DS}/holdout.json").read_text())
coco = json.loads(Path(f"{DS}/coco.json").read_text())
print(f"holdout: {len(hold)} frames")
```

## Celda 4 — corrida A: tcd Mask-RCNN (NO SegFormer por licencia)
```python
# tcd-predict semantic|instance <model> <img> <out>
# Usar variante Mask-RCNN/U-Net. Ejemplo por frame del holdout:
# !tcd-predict instance restor/tcd-mask-rcnn /kaggle/input/.../frame.png /kaggle/working/pred_tcd/
```
Ajustar el nombre exacto del modelo al zoo vigente (`ZOO.md` del repo tcd).

## Celda 5 — corrida B: DeepForest (cajas)
```python
from deepforest import main
m = main.deepforest()
m.use_release()  # pesos pre-entrenados
# predecir por frame del holdout -> boxes CSV
```

## Celda 6 — métricas vs oro (pycocotools)
- mAP50-box (ambos), mAP50-mask (tcd), MAE de conteo por frame, tiempo/frame.
- Guardar `/kaggle/output/baseline_report.json` + 3 overlays por finca.

## Gate de decisión
- mAP50-mask tcd > 60% → fine-tune corto desde esos pesos (Fase 2).
- 40-60% → fine-tune completo con los 331 oro.
- <40% → entrenar desde cero; el dominio es raro y ya lo sabemos con número.
