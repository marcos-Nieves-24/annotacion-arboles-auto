# Pipeline completo de anotación automática de copas — 2026-11-19

Este documento describe **todo** lo construido: desde la búsqueda de alternativas hasta los dos frames subidos a CVAT. Incluye los dos pipelines que compiten (v4 y StarDist), sus resultados, y por qué se adoptó cada decisión.

---

## 1. Objetivo y restricciones

Anotar automáticamente copas individuales de limón en imágenes RGB de dron (tiles 512×512), donde las copas se tocan y los pipelines anteriores (SAM2 automático, clasificador de textura) fallaban.

Restricciones del proyecto:
- Todo entrenamiento en **Kaggle T4**; en local solo inferencia liviana, `foreground`, con `timeout`, sin procesos residuales.
- **CVAT solo lectura** salvo autorización explícita por lote.
- Sin subir credenciales al repo; la key se lee desde `scripts/ronda1.py`.
- Sin LiDAR, multiespectral ni cámara de altura: solo RGB.

---

## 2. Fase 0 — Búsqueda de alternativas

`docs/PLAN_BUSQUEDA_ALTERNATIVAS.md`. Se revisaron OpenAlex, Semantic Scholar, arXiv, GitHub y editoriales. Hallazgos principales:

- **FG-TreeSeg** (2026): training-free, separa copas que se tocan por flujo de Cellpose-SAM. Descartado inicialmente por costo; quedó como plan B.
- **StarDist** (Tong & Zhang, RSE 2025): F1 0.88 en bosque mixto, >92% accuracy, usa polígonos star-convex. Quedó como plan A.
- **HTC + CB-ResNet50** (concurso ISPRS 2025): AP50 0.41, AP75 0.21.
- **RSPrompter / BalSAM** (NeurIPS 2025): aprenden prompts para SAM.
- **SAM-M2F** (CVPRW 2025): extensión de etiquetas con SAM 2.

---

## 3. Fase 1 — Modelo de centroides (v1)

### 3.1 Arquitectura y targets

`docs/MODELO_CENTROIDES_V1.md`, `data/centroides/v1/`.

| Elemento | Detalle |
|---|---|
| Entrada | RGB 512×512 |
| Backbone | ResNet-18 ImageNet + FPN stride 4 (64 canales) |
| Cabezas | Heatmap (Conv 1×1 → logits) + radio (Conv 1×1 → softplus) |
| Target calor | Disco binario del núcleo Gaussiano (`σ = clamp(0.25·r, 2, 16)/4`, umbral ≥ 0.5) |
| Target radio | `r = √(area/π)` en píxeles del núcleo; borde del tile con peso 0.5 |
| Centroide | Shoelace sobre `segmentation` (el campo `bbox` del oro está en 0) |

### 3.2 Entrenamiento

- Kernel: `marcosalvarez234/limon-centroides-v1` (versión 5), dataset `limon-gold-v2d`, T4.
- Pérdida calor: BCE con `pos_weight=100`. Pérdida radio: SmoothL1 — **tenía un bug**: comparaba contra `torch.zeros_like(pred)` y hacía broadcasting `(B,1,H,W)` vs `(B,H,W)` → la cabeza colapsó a ~0.03. Corregido después en v2/v3.
- Optimizador AdamW, lr 5e-4, cosine; batch 16; 80 épocas; augment flip/rot/escala/HSV; semilla 7.
- 346 train, 38 dev, 20 holdout, 9 auditados excluidos.
- Resultado: train/dev/holdout recall 0.843/0.826/0.827 (no memoriza).

Bug corregido y documentado en `docs/REENTRENAMIENTO_RADIO.md`.

### 3.3 Métricas v1

| Métrica | Valor |
|---|---:|
| Mejor época / val loss | 39 / 0.281 |
| F1 cascada dev / holdout (v1.1) | 0.924 / 0.894 |
| Precisión holdout | 0.919 |

---

## 4. Fase 2 — Cascada v1.1 / v1.2 (la adoptada)

### 4.1 Pipeline completo

```
tile RGB 512×512
   │
   ▼
ResNet-18 + FPN → heatmap → picos = centroides (cx, cy)
   │
   ├─ radio = 2.17·√(N_{heat>0.5}/π)·4   (blob del heatmap)
   │  + vecino/2 como fallback post-NMS
   │  + recorte [0.6, 1.6]·mediana de la finca
   │
   ▼
SAM ViT-B con box ±1.15·r → 3 máscaras → se elige por score
   │
   ▼
Validador:
   área [0.20, 2.0]·π·r²   (calibrado en 43 rechazos de frames revisados)
   circularidad ≥ 0.20
   solidity ≥ 0.65
   NMS de máscara IoU ≥ 0.5
   conteo: poda si > 1.3×esperado hasta 1.1×+1
   │
   ▼
aceptadas (máscaras silver) / rechazadas → cascade_review.json
   │
   ▼
v4: detector de ambigüedad — pares IoU>0.15 sin surco claro → baja confianza
```

### 4.2 Calibración del validador (v1.2)

`docs/CALIBRACION_VALIDADOR.md`. Se usó CVAT **solo offline**, en frames con revisión humana (Villas 0–199, Magdala 0–130):

- 29 de 43 rechazos (67.4%) eran copas reales.
- Features de 223 propuestas: bajar piso de área `0.35 → 0.20` y circularidad `0.30 → 0.20` recupera 16 copas reales con precisión casi igual.
- Resultado Kaggle: v1.1 dev 0.9241/holdout 0.8938 → v1.2 dev 0.9332/holdout 0.8998, FP 25/17. Adoptada.

### 4.3 v4 — detector de ambigüedad

`docs/CALIBRACION_VALIDADOR.md` §7. Para pares con IoU > 0.15, mide contraste de vegetación en el surco entre centroides vs extremos; si contraste < 0.15, marca baja confianza en `ambiguous_pairs.json`. No parte ni borra nada. F1 idéntico a v1.2; 11/4 pares marcados en dev/holdout.

---

## 5. Fase 3 — Experimentos de intersección (todos descartados)

| Experimento | Resultado | Por qué se descartó |
|---|---|---|
| Voronoi global (`cascade-v30`/`v32`) | IoU 0 pero área ratio 0.64 | Cortaba lóbulos reales |
| Selección informada de máscara (`v31`) | F1 0.877 | Perdía copas |
| Priority painting (`v30`) | IoU 0, area 0.99 | **No separaba**: 16 máscaras → 10 componentes |
| Recalibración del radio (v2-radio, v3-head, v2.1) | radio hasta 93 px | Cascada peor que v1.1 (0.872 vs 0.894) |
| A+B, blend, max, boost (`test100/barrido_radio.json`) | ningún gana en todos los bins | Compresión estructural del blob |
| Flag de vegetación | grandes 5%→10.5% | Insuficiente |

---

## 6. Fase 4 — StarDist (opción A, conservada)

> Nota 2026-11-19: en su momento no se adoptó porque no pasó los gates E2. El usuario corrigió esa decisión: **StarDist no se descarta**, se conserva como pipeline y se evalúa por finca contra `gold_v2` (ver `docs/MATRIZ_CONFUSION_STARDUST.md`).

`docs/PROPUESTAS_PRECISION.md`, `docs/PLAN_CASCADE_V5.md`.

### 6.1 Por qué se probó

Test propio: 576 copas del oro reconstruidas con 36 rayos → IoU mediano 0.981, 99.8% ≥ 0.8. Las copas son star-convex, justo la suposición de StarDist. Literatura: F1 0.88–0.92, NMS de solape ajustable a ~0.

### 6.2 Entrenamiento

- Kernel `marcosalvarez234/limon-stardist-v1`, dataset `limon-gold-v2d`, T4.
- Etiquetas: 3774 copas no-iscrowd, 21 en cuarentena excluidas, `fill_label_holes`, normalización `csbdeep normalize(1, 99.8)`.
- `Config2D`: `n_rays=32`, `grid=(2,2)`, `n_channel_in=3`, U-Net depth 3, patch 256, batch 8, 120 épocas × 100 pasos, lr 3e-4, `foreground_only=0.9`, `dist_loss=mae`, `loss_weights=[1,0.2]`.
- 74 minutos. Error final al convertir `optimize_thresholds` (devuelve objeto `Thresholds`, no dos floats). Pesos recuperados.

### 6.3 Problemas de infraestructura

- Un kernel con `ERROR` no sirve como `kernel-source` → se subieron los pesos como dataset `limon-stardist-v1-weights`.
- Kaggle descomprime automáticamente los zips de dataset.
- El builder sobrescribe el metadata: hay que pasar todos los `--dataset` y luego arreglar el JSON.

### 6.4 Evaluación

Kernel `marcosalvarez234/limon-stardist-eval` (v6). Resultado:

| Métrica | v1.2 | StarDist |
|---|---:|---:|
| F1 dev | 0.9332 | 0.8567 |
| F1 holdout | 0.8998 | 0.8372 |
| Pares IoU > 0.05 | 39 | **0** |
| Circularidad | ~0.57 | 0.87 |

StarDist elimina el solape de píxeles pero las copas quedan redondas y **pegadas físicamente**. Test en frames 261–270 (`data/stardist_261_270/`): 16 máscaras en el 262 pero solo 10 componentes conexos.

**Evaluación posterior (2026-11-19):** se corrió sobre las 405 imágenes de `gold_v2` con matriz de confusión por finca y split → `docs/MATRIZ_CONFUSION_STARDUST.md`, datos en `data/stardist_confusion/`. Resultado honesto (holdout, NMS 0.1): F1 0.835 global, **0.860 en LasVillas vs 0.807 en Magdala**. Anotación del rango 262-300 en `docs/STARDUST_262_300.md`.

---

## 7. Fase 5 — Subidas a CVAT

| Frame | Pipeline | Formas nuevas | Estado |
|---|---|---:|---|
| 261 (`00771`) | cascada v4 (adoptada) | 12 | subido, HTTP 200, 6→18 formas |
| 262 (`00772`) | StarDist NMS 0.02 (descartado) | 16 | subido, HTTP 200, 4→20 formas |

Método: `PUT` a `/api/jobs/4307835/annotations` con todas las formas existentes (sin `id`) + las nuevas. Backup antes de cada subida en `data/para_subir/villas_26{1,2}/backup_annotations_before_upload.json`. `PATCH` devuelve 400 ("Please specify a correct action"). Etiqueta `crop_tree`, `label_id=8155070`.

---

## 8. Estado actual y decisiones

- **Conservado y en uso**: StarDist. Pipeline completo en raíz (`kernels/stardist_*`), anotó 24 frames del 262-300 (`docs/STARDUST_262_300.md`) y fue evaluado por finca (`docs/MATRIZ_CONFUSION_STARDUST.md`). No se descarta.
- **Adoptado históricamente**: cascada v4 (v1.2 + detector de ambigüedad). F1 holdout 0.900, con solapes parciales y cola de revisión. Quedó en `experimentos/cascada/`.
- **Pendiente de explorar**: FG-TreeSeg (flujo), opción B de `docs/PROPUESTAS_PRECISION.md`.

## 9. Índice de artefactos

| Archivo | Contenido |
|---|---|
| `odd/tasks/centroides.md` | Log ODD completo (S1–S19, T1–T19) |
| `docs/MODELO_CENTROIDES_V1.md` | Modelo v1 congelado |
| `docs/CALIBRACION_VALIDADOR.md` | Calibración v1.2 y v4 |
| `docs/REENTRENAMIENTO_RADIO.md` | Bug de `radius_loss` y corridas v2/v3 |
| `docs/PROPUESTAS_PRECISION.md` | Revisión bibliográfica y propuestas A–D |
| `docs/STARDUST_262_300.md` | Anotación StarDist de 24 frames (262-300) y reproducibilidad |
| `docs/MATRIZ_CONFUSION_STARDUST.md` | Matrices de confusión por finca y split contra `gold_v2` |
| `docs/PLAN_CASCADE_V5.md` | Plan no implementado de priority painting |
| `data/centroides/v1/` | Modelo congelado + código + métricas |
| `data/centroides/cascade_v11/`, `v30/`–`v41/` | Experimentos de cascada |
| `data/centroides/test100/`, `test_261_270*/` | Matrices y barridos |
| `data/stardist_v1/`, `stardist_eval/`, `stardist_261_270/` | StarDist |
| `data/para_subir/villas_261/`, `villas_262/` | Paquetes y backups CVAT |
