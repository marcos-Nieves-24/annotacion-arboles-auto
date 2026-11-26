# Calibración del validador con frames revisados — 2026-11-19

## Principio

CVAT se usa **solo offline para calibrar**, nunca en inferencia. El pipeline en producción no consulta CVAT: predice centroide + radio, aplica umbrales congelados y produce máscaras o rechazos para revisión.

## 1. Medición del problema

Sobre los rechazos de `cascade_v11` en frames con revisión humana (Villas 0–199, Magdala 0–130):

| Motivo | Rechazos | Copas reales (CVAT revisado) |
|---|---:|---:|
| `area` | 26 | 19 (73%) |
| `conteo` | 15 | 9 (60%) |
| `forma` | 2 | 1 (50%) |
| **Total** | **43** | **29 (67.4%)** |

Antes se comparaba contra `gold_v2`, que es un subconjunto filtrado: por eso esas copas parecían falsos positivos. Son copas reales.

## 2. Calibración offline

Se extrajeron features de 223 propuestas en esos frames (172 reales, 51 no reales): `area_ratio = área_SAM/(π·r²)`, circularidad y solidity.

| Regla | TP | FP | FN | Precisión | Recall |
|---|---:|---:|---:|---:|---:|
| Actual `[0.35, 2.0]`, circ ≥ 0.30, sol ≥ 0.65 | 152 | 43 | 20 | 0.779 | 0.884 |
| Calibrada `[0.20, 2.0]`, circ ≥ 0.20, sol ≥ 0.65 | 168 | 49 | 4 | 0.774 | **0.977** |

Recupera 16 copas reales con precisión casi idéntica. El límite superior de área y `solidity` no aportaron: el cambio útil es el piso de área y la circularidad.

## 3. Validación en Kaggle

Dos kernels nuevos sobre la base v1.1 (misma arquitectura y pesos v1), con `accelerator: T4`:

| Kernel | Umbrales | F1 dev | F1 holdout | P/R holdout | FP dev/holdout |
|---|---|---:|---:|---|---|
| `cascade-v11` | original | 0.9241 | 0.8938 | 0.919/0.870 | 23/16 |
| `cascade-v40` | calibrado + conteo activo | **0.9332** | 0.8998 | 0.915/0.885 | 25/17 |
| `cascade-v41` | calibrado + conteo solo aviso | 0.9219 | **0.9104** | 0.917/0.904 | 40/17 |

Rechazos: v11 `{area:27, conteo:15, forma:2}` → v40 `{area:6, conteo:24}` → v41 `{area:6}`.

## 4. Decisión

**Adoptar `cascade-v40`** como v1.2 canónica.

- Mejora F1 en **ambos** splits (+0.009 dev, +0.006 holdout) con un costo mínimo: +2 FP en dev, +1 en holdout, precisión −0.003 en ambos.
- `v41` da mejor F1 en holdout pero **pierde 3.8 pp de precisión en dev** (23 → 40 FP) al desactivar la poda por conteo. Esa mejora puede ser ruido de unas pocas copas; el costo en dev es real. Queda documentada como alternativa si en el futuro se prioriza recall.
- Los gates E2/E3 pasan en las tres versiones.

## 5. Notas

- Un solo run por versión: las diferencias son chicas y no se repitieron con otra semilla. La adopción de v40 se basa en que es la única que mejora los dos splits sin degradar precisión.
- El conteo sigue siendo alarma blanda con poda (recorta hasta `1.1×esperado + 1` cuando supera `1.3×`), pero ahora recorta menos porque el filtro de área ya no tira copas reales por error.
- Los pesos del modelo de centroides no cambiaron: `data/centroides/v1/model.pt`, sha256[:16] `ec2bc282e40bd94a`.

## 7. v4 — detector de ambigüedad y regla de confianza (2026-11-19)

Sobre v1.2, sin cambiar pesos ni máscaras: para cada par de máscaras aceptadas con **IoU > 0.15**, se mide el contraste del surco entre centroides (vegetación en la franja media vs vegetation en los extremos). Si el contraste < 0.15 se marca el par como **baja confianza** y va a `ambiguous_pairs.json` para revisión humana. **No se parte, no se borra y no se fusiona nada**: es una etiqueta de confianza, no un cambio geométrico.

| | F1 dev | F1 holdout | Pares que se tocan | Pares ambiguos |
|---|---:|---:|---:|---:|
| v1.2 (v40) | 0.9332 | 0.8998 | — | — |
| v4 | **0.9332** | **0.8998** | 13 dev / 6 holdout | 11 dev / 4 holdout |

Gates E2/E3 pasan, sin cambios: es esperable, porque el postproceso no modifica ninguna máscara. El valor de v4 es la **cola de revisión priorizada**: 15 pares de 19 que se tocan no tienen un surco detectable y se entregan marcados.

**Lectura honesta**: el detector es conservador — marca el 79% de los pares que se tocan. En varios casos el "contraste" sale negativo (la franja entre copas es *más verde* que los extremos, probablemente por maleza intermedia). Sirve como bandera de revisión, no como clasificador de certeza. Si se quiere menos ruido en la cola, hay que subir el umbral de IoU o bajar el de contraste y volver a medir; no se hizo en esta entrega.

Artefactos: `data/centroides/cascade_v4/` (reporte, `ambiguous_pairs.json`, overlays con círculo cian para baja confianza). Kernel `limon-centroides-cascade-v4`.

## 8. Artefactos

- `data/centroides/calibracion/rechazos_revisados.json`, `features.json`.
- `data/centroides/cascade_v40/`, `cascade_v41/` con reportes, reviews y overlays.
- Kernels: `limon-centroides-cascade-v40`, `limon-centroides-cascade-v41`.
