# Plan — Clasificador de parches por textura (copa vs fondo)

Basado en Rossi et al. (IGARSS 2026, DOI 10.1109/IGARSS60577.2026.11730742).
Objetivo: rescatar las copas que YOLO11n-seg (T1b) no detecta (grupo B de la auditoría), usando clasificación de parches con textura, sin depender del box del detector.

## 1. Principio del paper aplicado

El hallazgo accionable NO es Swin, es el **copy-paste de textura propia**: rellenar el fondo del parche con textura muestreada de la misma copa, para que el clasificador no aprenda atajos de fondo negro ni de forma del polígono. En nuestro RF viejo pasó lo contrario: aprendió a separar por área/fondo y dio AP 0.9997 engañoso.

El paper reporta: ResNet-18 F1 0.93, Xception 0.93, ViT 0.86, Swin 0.96. Empezamos con ResNet-18 (liviano, casi igual que Swin en datos chicos).

## 2. Dataset de parches

### Positivos
* 3774 copas `iscrowd=0` del `gold_v2/coco.json` (segmentación real).
* Recorte: bbox de la máscara → cuadrado, padding a **640×640** centrado.
* Relleno del fondo: copy-paste de textura muestreada de la **misma copa** (papel §II-C), no negro.
* Augment: flip horizontal/vertical, rotación leve, jitter HSV — igual que el paper.

### Negativos
* 27 frames sin copas (`gold_v2` tiene 405 frames, 378 con copas).
* Parches de **suelo/camino/sombra** recortados de esos frames + de zonas fuera de las copas en frames con copas.
* Minería de hard negatives: los 9 frames que el usuario auditó como "copas reales no detectadas" — recortes centrados en esas copas sirven como positivos difíciles.

### Split (evitar fuga — lección aprendida)
* Split por **frame**, no por parche: todos los parches de un frame van al mismo lado.
* 70/30 train/val, seed fijo. Frames del dev y holdout del experimento YOLO quedan **excluidos de train** del clasificador (para no contaminar la medición posterior).
* Dev/holdout del clasificador = los 9 frames auditados + frames vecinos de terreno, separados.

## 3. Modelo

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Arquitectura | ResNet-18 (timm/torchvision) | F1 0.93 en el paper, 11M params, corre en T4 y CPU |
| Input | 640×640×3 | igual paper |
| Clases | 2 (copa / fondo) | binario, no especies |
| Pretrained | ImageNet | transfer learning |
| Loss | BCEWithLogits | binario |
| Optimizer | AdamW lr 1e-4 | fine-tune (lr bajo) |
| Epochs | 30, patience 5 | datos chicos, early stopping |
| Batch | 32 | — |
| Augment | flip, rot ±15°, HSV, copy-paste textura | papel §II-D |
| Seed | 7 | reproducible |

Si ResNet-18 no llega a F1 ≥ 0.90 en val, subir a **Swin-T** (el paperusa Large; nosotros Tiny por memoria).

## 4. Uso para rescatar copas (inferenciaslide-window)

1. Sobre los 9 frames auditados (y cualquier frame con bajo conteo vs esperado), barrer ventanas deslizantes de 640×640 con stride 320 sobre la imagen.
2. Cada ventana → clasificador → probabilidad de "contiene copa".
3. Agrupar ventanas con prob > 0.5 por solapamiento (NMS de ventanas).
4. Para cada ventana positiva, proponer un box candidato (centro de la ventana) y pasar al **segmentador** (YOLO mask existente o SAM refine —— decisión después) para obtener la máscara.
5. **No se agrega automáticamente**: candidato va a revisión humana (`RESCUE_REVIEW`).

## 5. Métricas y criterios

| Métrica | Cómo se mide |
|---|---|
| F1 del clasificador | en val de parches (copa vs fondo) |
| Recall rescue | cuántas de las copas de los 9 frames auditados se recuperan |
| FP introducidos | cuántas ventanas positivas caen sobre suelo |
| Impacto final | instance F1 global en holdout, antes vs después del rescate |

**Aceptación**: el rescate recupera ≥50% de las copas perdidas del grupo B sin bajar la precisión global más de 2pp. Si no, se descarta igual que SAM.

## 6. Infraestructura

* Entrenamiento: local CPU (ResNet-18, 3774 parches, ~30 ep es viable) o Kaggle T4 si es lento. No requiere GPU nueva.
* Datos: `data/gold_v2/`, imágenes en `data/r1/images/` y `data/gold_v1/images/`.
* Sin CVAT writes (regla dura: cero subidas sin autorización explícita por lote).

## 7. Orden

1. Generar parches (positivos con copy-paste, negativos) → `data/fase4/patches/`.
2. Entrenar ResNet-18 → medir F1 en val.
3. Slide-window sobre los 9 frames → medir recall rescue y FP.
4. Decidir: integrar como capa de rescate o descartar.

## Log
- Basado en paper leido en `docs/articles/` (pdftotext), 2026-11-18.
- ResNet-18 elegido sobre Swin por simplicidad; Swin es plan B.
- El truco del copy-paste es el hallazgo clave, no la arquitectura.
