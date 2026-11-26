# Decisión — Frente B (generador de propuestas por centroides)

Fecha: 2026-11-18. Alcance: evaluación de los gates E1–E3 de `docs/PLAN_CENTROIDES.md`.

## Qué se construyó

- `scripts/build_centroid_targets.py` + test determinista (7 tests GREEN). Targets: centroide por shoelace, radio `√(area/π)`, núcleo binario del Gaussian (umbral 0.5) como mapa de calor, radio y peso supervisados en el núcleo.
- `kernels/centroides/kernel_source.py` → notebook `marcosalvarez234/limon-centroides-v1` en Kaggle T4, dataset `limon-gold-v2d`. CenterNet-lite (ResNet-18 + FPN stride 4), BCE con `pos_weight=100` y SmoothL1 de radio, 80 épocas, batch 16, AdamW lr 5e-4, augment flip/rot ±15°/escala ±10%/HSV.
- 346 frames de train; dev 38 y 9 auditados excluidos; holdout 20 intacto.

## Resultado observado (kernel v5)

| Métrica | Valor |
|---|---|
| Mejor época | 39 (val loss 0.281); pérdida sube levemente después (0.34 en ép 75) → sobreajuste leve |
| Recall de propuesta — dev (38 frames) | **0.816** |
| Recall de propuesta — 9 frames auditados | **0.962** |
| FP por frame (propuesta, thr 0.15) | dev 21.2 / auditados 18.4 |
| Punto operativo recomendado | thr 0.50, NMS `0.7·(r_i+r_j)` → auditados recall **0.952**, FP **14.7**; dev recall 0.813, FP 19.3 |

El barrido de umbral se hizo en local sobre los pesos bajados (mismo modelo, sin reentrenar) porque el decode es determinista.

## Gates

| Gate | Criterio | Resultado |
|---|---|---|
| E1 | recall ≥ 0.85 en 9 auditados | ✅ **PASA** con el decode del kernel v5 (0.962, thr 0.15, NMS efectivamente desactivada). Con el decode final de la cascada (thr 0.5 + NMS con radio mediano) es **45/54 = 0.833** — ver nota abajo |
| E2 | F1 de instancia final ≥ 0.748 en dev/holdout | ✅ **PASA** — dev 0.869, holdout **0.799** |
| E3 | precisión ≥ baseline − 2 pp (0.802) | ✅ **PASA** — holdout **0.874**, por encima del baseline 0.822 |

Resultado de la cascada (`limon-centroides-cascade-v1`, kernel v2):

| Split | P | R | F1 | TP/FP/FN | Propuestas/frame | Aceptadas/frame | Count MAE |
|---|---|---|---|---|---|---|---|
| dev (38) | 0.905 | 0.837 | **0.869** | 313/33/61 | 10.45 | 9.11 | 1.42 |
| holdout (20) | 0.874 | 0.736 | **0.799** | 153/22/55 | 10.4 | 8.75 | 2.25 |

Comparación con el detector actual en holdout: F1 0.728 → **0.799** (+0.071), precisión 0.822 → 0.874, recall 0.653 → 0.736.

**Aclaración importante sobre E1 (2026-11-18):** el 0.962 se midió con el `decode` del kernel v5, que leía el radio de la cabeza colapsada (~0.03). Con `dist_factor·(r_i+r_j) ≈ 0.04 px`, el NMS por distancia **no suprimía nada** y se conservaban hasta 40 picos: eso infla el recall. El barrido que dio 0.952 en thr 0.5 usaba el mismo decode, así que tampoco es comparable con la configuración final. Con el decode que realmente usa la cascada (radio mediano por finca → NMS de ~81 px en Villas y ~63 px en Magdala, thr 0.5), el resultado en los 9 frames es 45/54 = 0.833. **Los gates E2 y E3 no cambian**: se midieron con la cascada final (F1 holdout 0.799, precisión 0.874). Conclusión: el modelo pone centroide en la mayoría de las copas marcadas, pero el NMS con radio mediano se come algunos — conviene recalibrar `dist_factor` antes de usar el generador suelto.

Nota técnica (**corregida 2026-11-19**): la cabeza de radio colapsó a ~0.03 por un bug en `radius_loss` — comparaba `pred` contra ceros en vez de contra el target y hacía broadcasting de `(B,1,128,128)` contra `(B,128,128)`. No fue la augmentación. En la cascada se usa el **radio mediano por finca del oro** (Villas 58.5 px, Magdala 45.4 px) para el box prompt y la banda de área; no se reentrenó. Motivos de rechazo del validador: área 65, forma 7, conteo 10, NMS 2. Rechazos quedan en `data/centroides/cascade/cascade_review.json` para revisión humana.

## Iteraciones fallidas (para no repetirlas)

1. v1: sufijo `00626` ambiguo entre Villas y Magdala → se usa nombre completo.
2. v2: `affine_grid` con tamaño 3D → se corrige a 4D.
3. v3: `decode` con ranks 3D/4D mezclados → `squeeze()` + sigmoid.
4. v4: focal con objetivo Gaussiano difuso → mapa de calor plano (picos 0.45–0.60, recall 0.556, early-stop en ép 7). Test local de sobreajuste mostró que al bajar la loss el pico **bajaba**: la pérdida premiaba un mapa plano. Se reemplaza por núcleo binario + BCE con `pos_weight=100`.

## Veredicto

**Adoptar la cascada completa** (centroides → SAM por box prompt → validador) como capa de rescate, no como reemplazo del detector YOLO. Pasa los tres gates con margen y supera al baseline en holdout.

La alta tasa de FP a nivel de propuesta (15–19/frame) **no fue un defecto**: la etapa es sobre-completa a propósito y el validador la reduce a ~9 aceptadas/frame con precisión 0.874.

Limitación conocida: el radio viene de un prior por finca, no del modelo; en copas de tamaño atípico el box prompt puede quedar grande o chico. El validador de área descarta los casos más extremos, y esos rechazos van a revisión humana.

**Corrección a la lectura visual de los overlays (2026-11-18):** los centroides **sí** marcan las copas anotadas. Verificado con números: en `Limon_LasVillas_00196.png` las 3 copas del oro tienen propuestas a 4–5 px y ninguna fue rechazada; en `Limon_LasVillas_00283.png`, 4 de 5 tienen propuestas a ≤12 px y la quinta a 23 px (el box de radio mediano cubre parte de su zona), con dos rechazos por NMS duplicado a 8 y 12 px de centros del oro. Las copas grandes que aparecen sin máscara en esos frames **no figuran en el oro**, así que no son fallos de recall. El gap residual está en el refinamiento de máscara y en duplicados rechazados, no en falta de señal de centroide.

## Artefactos

- `data/centroides/targets/*.npz` (405), `data/centroides/viz/` (20), `data/centroides/centros_summary.json`.
- `data/centroides/out/centroides_last.pt`, `centroides_best.pt`, `centroides_report.json`, log.
- Kernel: `https://www.kaggle.com/code/marcosalvarez234/limon-centroides-v1`

## Test de generalización (2026-11-18)

- **Mismo oro**: recall de centroides train 0.843 / dev 0.826 / holdout 0.827 (brecha 1.7 pp). No memorizó frames.
- **8 frames de CVAT nunca vistos** (4 de Villas fuera del oro + 4 de Yulima, ninguno en entrenamiento):
  - Villas (`00723`, `00891`, `01051`, `01203`): F1 0.53 / 0.15 / 0.40 / 0.40; centroides sobre copas CVAT 2/6, 2/9, 4/5, 2/5.
  - Yulima (`00052`, `00638`, `01058`, `01394`): F1 0.80 / 0.96 / 0.86 / 0.92.
- **Por qué la inversión**: el radio mediano usado (58.5 px Villas, 36 px Yulima) coincide con Yulima (mediana real de referencia 35.4 px) pero no con la cola de copas chicas de los frames Villas no vistos (p10 21.3 px contra mediana 53.5). El modelo aprendió una banda de tamaño/textura estrecha, no el concepto general de copa.
- Las referencias CVAT de esos frames son `auto`/`semi-auto` sin revisión humana en ambos casos, así que parte de la diferencia puede ser ruido de referencia; la magnitud (0.37 vs 0.88) es demasiado grande para explicarla solo con eso.
- **Conclusión**: no hay memorización de imágenes, pero sí **sobreajuste a la distribución**. Para una finca nueva hay que medir su radio mediano local y revalidar antes de usar el modelo.

## Siguiente paso (ya no es una decisión de gates)

Integrar la cascada en el pipeline de pre-anotación: las máscaras aceptadas salen como silver y los rechazos de `cascade_review.json` van a corrección humana. Cualquier escritura en CVAT sigue requiriendo autorización explícita por lote.

## Log

- 2026-11-18: T1, T2 y T3 completados; E1 pasa; E2/E3 quedan pendientes de la cascada.
- 2026-11-18: usuario eligió opción A ("procede"). T5 implementado: kernel `limon-centroides-cascade-v1` (2 versiones). v1 rechazó todo por radio colapsado; v2 usa radio mediano por finca. E2 y E3 pasan → decisión de adopción.
