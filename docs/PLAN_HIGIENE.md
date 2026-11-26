# Plan de higiene del repositorio — 2026-11-19

Objetivo: dejar el workspace limpio alrededor del **pipeline StarDist** (el que se quiere conservar), moviendo todo lo demás a una carpeta de experimentos. **No se borra nada todavía**: este es el plan.

## 1. Qué se conserva en raíz (pipeline StarDist)

### Código
```
kernels/stardist_v1/kernel_source.py          # entrenamiento
kernels/stardist_eval/kernel_source.py        # evaluación
kernels/stardist_261_270/kernel_source.py     # inferencia + export CVAT
scripts/build_kernel_centroides.py            # builder (lo usan los 3)
```

### Metadata de kernels (crítica: el builder la sobrescribe)
```
kernels/stardist_v1/kernel-metadata.json
kernels/stardist_eval/kernel-metadata.json
kernels/stardist_261_270/kernel-metadata.json
```

### Datos y pesos
```
data/stardist_v1/limon_stardist_v1/   # weights_best.h5, config.json, thresholds.json
data/stardist_eval/stardist_report.json
data/stardist_261_270/stardist_261_270_report.json
data/stardist_261_270/cvat_262_stardist.json
data/para_subir/villas_262/            # paquete subido a CVAT (manifest, annotations, backup)
```

### Documentación
```
docs/PIPELINE_COMPLETO.md             # sección 6 = StarDist
docs/PROPUESTAS_PRECISION.md          # por qué se probó
```

### Imágenes de entrada (compartidas, se quedan)
```
data/gold_v2/          # 4.6 MB — coco.json + splits, usado en entrenamiento/eval
data/raw/              # 36 MB — anotaciones CVAT crudas
data/r1/               # 1.3 GB — imágenes fuente
data/kaggle-villas-261-270/   # 6 MB — copia local de las 10 imágenes
```

## 2. Qué se mueve a `experimentos/`

Crear esta estructura:

```
experimentos/
├── cascada/                 # centroides + SAM + validador (v1.1 → v4)
│   ├── kernels/             # kernels/centroides*, kernels/centroides_cascade*
│   ├── data/                # data/centroides/ completo
│   ├── scripts/             # scripts de prepare/upload de centroides
│   └── docs/                # PLAN_CENTROIDES, DECISION_CENTROIDES, CALIBRACION_VALIDADOR,
│                            # MODELO_CENTROIDES_V1, RADIO_COPA_ESTIMACION,
│                            # REENTRENAMIENTO_RADIO, PLAN_CASCADE_V5
├── yolo/                    # todo el trabajo anterior de YOLO/occ/clasificador
│   ├── kernels/             # baseline-fase1, bench-nmm, fase2*, fase4*, occ*, ronda1b, texture-classifier
│   ├── data/                # kaggle-out*, weights-r1, mejora, occ, occ-clean, profile, fase4, gold_v1, gold_v2_raw
│   ├── scripts/             # scripts de esos experimentos
│   └── docs/                # INFORME_2026-11-11, PLAN_234, PLAN_FRENTE_A,
│                            # PLAN_TEXTURE_CLASSIFIER, kaggle-baseline-runbook
└── cvat/                    # subida del frame 261 con cascada v4
    └── data/para_subir/villas_261/
```

**Regla**: solo código, reportes y pesos. **No se mueven overlays, contact sheets ni imágenes de visualización** de cascade/Voronoi/SAM — se descartan (ver §3).

### Detalle de `cascada/data/` — qué se guarda y qué no

| Se guarda | No se guarda (se descarta) |
|---|---|
| `centroides/v1/` (modelo congelado + config + métricas + código) | `centroides/cascade*/overlays/` (todas) |
| `centroides/v1/code/` (copia de scripts y notebooks) | `centroides/comparacion_*.png` |
| `centroides/v2_radio/`, `v3_head/` (pesos + reportes) | `centroides/heatmaps/`, `centroids_sam/`, `circulos*/`, `priority*/`, `test_261_270*/overlays/` |
| `centroides/cascade*/centroides_cascade_report.json` | Todos los contact sheets |
| `centroides/cascade*/cascade_review.json` | `centroides/yulima_unseen/overlays*/`, `overlays_azules/`, `solo_azules/` |
| `centroides/cascade*/ambiguous_pairs.json` | `centroides/flag_vegetacion/`, `calibracion/` (solo reportes chicos, sí se guardan) |
| `centroides/test100/*.json` (matrices y barridos) | `centroides/viz/` |
| `centroides/targets/` (21 MB, targets npz) | — |

### Duplicados contaminantes (se eliminan, no se mueven)

- **9 copias de `sam_vit_b_01ec64.pth`** de 358 MB cada una dentro de `data/centroides/cascade*/` → **3.2 GB**. Es un checkpoint público descargable; no es información propia.
- `data/stardist_eval/model/` y `data/stardist_eval/model_weights/` → copias duplicadas de los pesos que ya están en `data/stardist_v1/`.(se quedan solo las de `stardist_v1/`)
- `data/kaggle-stardist-weights/` → copia de subida del dataset, redundante con `data/stardist_v1/`.
- `data/fiftyone.log` → log de una app cerrada.

## 3. Qué se elimina directamente (sin mover)

| Elemento | Tamaño aprox | Razón |
|---|---:|---|
| `data/centroides/cascade*/sam_vit_b_01ec64.pth` × 9 | 3.2 GB | Checkpoint público duplicado |
| Overlays / contact sheets de cascade, Voronoi, SAM | ~60 MB | Pedido explícito: no conservar |
| `data/stardist_eval/model/`, `model_weights/` | 22 MB | Duplicados |
| `data/kaggle-stardist-weights/` | 21 MB | Duplicado |
| `data/fiftyone.log` | 8 KB | Residual |
| `data/centroides/viz/`, `priority*/`, `circulos*`, `heatmaps/`, `centroids_sam/`, `flag_vegetacion/` overlays | ~60 MB | Visualizaciones descartadas |

## 4. Orden de ejecución (cuando des el OK)

1. Crear `experimentos/cascada/`, `experimentos/yolo/`, `experimentos/cvat/`.
2. Mover kernels y scripts (usar `mv`, nunca copiar).
3. Mover data de cascada/yolo/cvat **sin overlays**: usar `rsync --exclude` o `find` con filtros.
4. Mover docs.
5. Eliminar duplicados y visualizaciones listadas en §3.
6. Verificar que los 3 kernels de StarDist siguen corriendo: re-chequear imports y metadata.
7. Actualizar `docs/PIPELINE_COMPLETO.md` con las nuevas rutas.

## 5. Verificación final esperada

Después de la higiene:

```
.Kaggle kernels (sin cambios)
docs/
├── PIPELINE_COMPLETO.md
├── PROPUESTAS_PRECISION.md
├── PLAN_HIGIENE.md            <- este
kernels/
├── stardist_v1/
├── stardist_eval/
└── stardist_261_270/
scripts/build_kernel_centroides.py
data/
├── gold_v2/  raw/  r1/  kaggle-villas-261-270/
├── stardist_v1/  stardist_eval/  stardist_261_270/
└── para_subir/villas_262/
experimentos/
├── cascada/
├── yolo/
└── cvat/
```

Espacio liberado estimado: **~3.4 GB** de 8.2 GB.

## 6. Log

- 2026-11-19: plan escrito tras revisar 8.2 GB de `data/`, 68 scripts, 27 kernels y 5455 PNG. No se movió ni borró nada todavía.
- 2026-11-19: **ejecutado**. Movidos 12 kernels de cascada, 14 de YOLO, 12 docs y 73 scripts. Datos movidos: `data/centroides/` → `experimentos/cascada/data/` (324 MB, sin PNG ni SAM), `villas_261` → `experimentos/cvat/data/`, 16 carpetas de YOLO → `experimentos/yolo/data/`. Eliminados 9 checkpoints SAM (3.2 GB), duplicados de pesos StarDist, `fiftyone.log` y todas las visualizaciones de `data/centroides/`. **Resultado: `data/` pasó de 8.2 GB a 1.7 GB.** Verificación: 3 kernels StarDist en raíz, pesos y reportes íntegros, reportes de cascada (v11, v4, etc.) y targets preservados en `experimentos/cascada/`.
