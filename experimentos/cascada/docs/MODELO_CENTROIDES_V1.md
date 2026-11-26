# Modelo de centroides v1 — documentación de la versión congelada

Fecha: 2026-11-18. Estado: **v1 archivada**. Kernel de entrenamiento: `marcosalvarez234/limon-centroides-v1` (versión 5, `COMPLETE`). Kernel de cascada: `marcosalvarez234/limon-centroides-cascade-v1` (versión 3).

## 1. Objetivo

Generar **propuestas de centroides** de copas individuales de limón en tiles RGB de 512×512, incluso cuando las copas se tocan. No produce máscaras: su salida es un heatmap con un pico por copa. Las máscaras las produce SAM aguas abajo y un validador decide qué se publica.

El problema que resuelve: YOLO11n-seg pierde copas chicas/jóvenes (41% del oro de train sin detección) y SAM2 automático fusiona copas que se tocan. El centro de la copa es observable aunque el borde no lo sea.

## 2. Pipeline completo

```
gold_v2 (coco.json, polígonos)
   │  shoelace → centroide, √(area/π) → radio, Gaussian ≥ 0.5 → núcleo binario
   ▼
targets (128×128 heatmap + radio + peso)
   │
   ▼
CenterNet-lite (ResNet-18 + FPN stride 4)  ── entrenado en Kaggle T4
   │  heatmap de 128×128 con un pico por copa
   ▼
decode: máximos locales → topK 40 → umbral 0.5 → NMS por distancia
   │  (cx, cy, score) + radio mediano de la finca
   ▼
SAM ViT-B con box prompt ±1.15·r → máscara candidata
   │
   ▼
validador: área → forma → NMS de máscara → conteo blando
   ├── aceptadas → máscaras silver
   └── rechazadas → cascade_review.json → revisión humana
```

## 3. Arquitectura

| Componente | Detalle |
|---|---|
| Backbone | ResNet-18, pesos ImageNet (`ResNet18_Weights.IMAGENET1K_V1`) |
| Cuello | FPN 4 niveles (stride 4/8/16/32), 64 canales por nivel, top-down con upsample nearest |
| Cabeza de calor | Conv2d(64→1, 1×1), salida **logits** |
| Cabeza de radio | Conv2d(64→1, 1×1) + softplus — **colapsó, no se usa** |
| Salida | 128×128×1 por cabeza (512/4) |

## 4. Construcción de targets

- **Centroide**: shoelace sobre `segmentation`. El campo `bbox` de `gold_v2` está en 0 en las 4602 anotaciones; usarlo daría todos los picos en (0,0).
- **Radio**: `r = √(area/π)`.
- **Heatmap**: núcleo del Gaussiano centrado en el centroide, `σ_full = clamp(0.25·r, 2, 16) px`, `σ_heat = σ_full/4`, píxeles con `exp(-d²/2σ²) ≥ 0.5` se ponen en 1. Núcleo binario, **no** Gaussiano difuso.
- **Radio/peso**: supervisados en los píxeles del núcleo; copas que tocan el borde del tile → peso 0.5.

## 5. Hiperparámetros

### Entrenamiento

| Parámetro | Valor | Motivo |
|---|---|---|
| Pérdida calor | BCE con logits, `pos_weight=100` | 16 384 celdas de fondo contra ~300 de copa; sin compensación la red diría siempre "no hay" |
| Pérdida radio | SmoothL1, peso 1.0 | Pérdida secundaria; la cabeza colapsó |
| Optimizador | AdamW, lr 5e-4, wd 1e-4, cosine | — |
| Batch / épocas | 16 / 80, sin early stopping | Mejor val en época 39; sobreajuste leve después |
| Augment | flip H/V, rot ±15°, escala 0.9–1.1, HSV ±0.015 | Evita memorizar posición/luz |
| Semilla | 7 | Reproducible |
| Datos | 346 train, 38 dev, 9 auditados excluidos, 20 holdout intacto | Los 9 auditados se sacaron aunque uno cayera en train |

### Decode y cascada

| Parámetro | Valor |
|---|---|
| `thr` / `topK` | 0.5 / 40 (máximo observado 33–35 copas por frame) |
| NMS de centros | distancia `0.7·(r_i+r_j)` |
| Radio por finca | Villas 58.5 px, Magdala 45.4 px, Yulima 35.9 px |
| Box SAM | `±1.15·r` |
| Validación área | `[0.35, 2.0]·π·r²` |
| Validación forma | circularidad ≥ 0.30, solidity ≥ 0.65 |
| NMS de máscaras | IoU ≥ 0.5, queda el de mayor score |
| Alarma de conteo | recorta si > 1.3× esperado hasta 1.1× + 1 |

## 6. Métricas

Archivo completo: `data/centroides/v1/metrics.json`.

| Evaluación | Resultado |
|---|---|
| Mejor época / val loss | 39 / 0.281 |
| Recall de propuesta train / dev / holdout | 0.843 / 0.826 / 0.827 |
| Recall 9 auditados (E1, decode v5 thr 0.15) | 0.962 — **con NMS efectivamente desactivada** |
| Recall 9 auditados (decode final thr 0.5) | 45/54 = 0.833 |
| Cascada dev | P 0.905 / R 0.837 / **F1 0.869** |
| Cascada holdout | P 0.874 / R 0.736 / **F1 0.799** (baseline YOLO 0.728) |
| Frames CVAT nunca vistos — Yulima (4) | F1 0.80 / 0.96 / 0.86 / 0.92 |
| Frames CVAT nunca vistos — Villas fuera del oro (4) | F1 0.53 / 0.15 / 0.40 / 0.40 |

Gates: E1 ✅ (con la salvedad del decode), E2 ✅ (dev y holdout), E3 ✅ (precisión 0.874 ≥ 0.802).

## 7. Limitaciones conocidas

1. **La cabeza de radio colapsó** (~0.03 de salida): **corrección 2026-11-19** — la causa real fue un bug en `radius_loss`, que comparaba la predicción contra `torch.zeros_like(pred)` en vez de contra el target, y además hacía broadcasting de `(B,1,128,128)` contra `(B,128,128)` en `(B,B,128,128)`. No fue la augmentación, como se dijo originalmente. Se usa el radio mediano de la finca en su lugar.
2. **Brecha train/dev mínima** (1.7 pp): no memoriza imágenes, pero sí sobreajusta a una banda de tamaño/textura. En Villas con copas chicas (p10 21.3 px) cae el recall.
3. **E1 inflado**: el 0.962 original se midió con `dist_factor·(r_i+r_j) ≈ 0.04 px` porque el radio venía de la cabeza colapsada; no suprimía duplicados y conservaba hasta 40 picos.
4. **Rechazos por área**: 65 de 84 en los 9 auditados. El radio prior sobredimensiona el box en copas chicas. Ver `docs/RADIO_COPA_ESTIMACION.md`.
5. **Círculo ≠ copa**: `r_enclose/r_eq` mediana 1.32, p90 1.93; un círculo de `1.15·r_eq` cubre la copa completa solo el 10% de los casos. SAM no está estrictamente limitado por el box, pero el prior importa.

## 8. Artefactos archivados

```
data/centroides/v1/
├── model.pt            # centroides_last.pt congelado, 45 MB, sha256[:16]=ec2bc282e40bd94a
├── config.json         # arquitectura, targets, hiperparámetros, decode, cascada
├── metrics.json        # entrenamiento, validación, generalización, gates, limitaciones
├── MANIFEST.json       # 12 archivos con sha256, tamaño y descripción
└── code/
    ├── scripts/
    │   ├── build_centroid_targets.py      # targets desde el COCO del oro
    │   ├── test_build_centroid_targets.py # 7 tests deterministas
    │   └── build_kernel_centroides.py     # generador de notebooks de Kaggle
    └── kernels/
        ├── centroides/         # kernel_source.py + centroides.ipynb + metadata (entrenamiento)
        └── centroides_cascade/ # kernel_source.py + cascade.ipynb + metadata (cascada)
```

No se incluyen los scripts de análisis de radio/generalización (fueron inline sobre los pesos congelados; sus resultados están en los docs) ni el checkpoint de SAM (se descarga en el kernel). El `MANIFEST.json` lo deja explícito.

Otros artefactos: `data/centroides/targets/` (405 npz), `data/centroides/out/` (reportes y pesos originales), `data/centroides/cascade/`, `data/centroides/yulima_unseen/`, visualizaciones en `data/centroides/heatmaps/`, `centroids_sam/`, `centroids_sam/` y `yulima_unseen/overlays*/`.

## 9. Reproducción

```bash
# entrenamiento (solo Kaggle T4)
python scripts/build_kernel_centroides.py
kaggle kernels push -p kernels/centroides

# cascada (Kaggle T4, usa el kernel anterior como kernel source)
python scripts/build_kernel_centroides.py \
  --source kernels/centroides_cascade/kernel_source.py \
  --out kernels/centroides_cascade \
  --id marcosalvarez234/limon-centroides-cascade-v1 \
  --title "Limon Centroides Cascade V1" \
  --code-file cascade.ipynb \
  --kernel-source marcosalvarez234/limon-centroides-v1

# targets y tests en local
PYTHONPATH=scripts .venv-verify/bin/python -m pytest scripts/test_build_centroid_targets.py -q
PYTHONPATH=scripts .venv-verify/bin/python scripts/build_centroid_targets.py --viz
```

Regla de infraestructura: todo entrenamiento en Kaggle; en local solo inferencia liviana, `foreground`, con `timeout`, sin procesos residuales.
