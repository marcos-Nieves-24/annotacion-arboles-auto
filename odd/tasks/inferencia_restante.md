# Inferencia StarDist en frames restantes — Villas y Magdala

Objetivo: extender la anotación StarDist a todos los frames sin inferencia de Villas y Magdala, con umbral 0.30. Yulima queda explícitamente fuera. Kaggle T4; en local solo checks y preparación; CVAT solo se escribe con autorización por lote.

## Specs

**S1** — Usuario: "prepara para los rangos sin inferencias". Preparar inferencia StarDist para los frames que no tienen salida de anotación: Villas 0-260 y 301-688 (649), Magdala 0-239 y 301-1301 (1241).

**S2** — Usuario: "c. no metas yulimar". Usar `prob_thresh=0.30` para Villas y Magdala; **no incluir Yulima** en ningún dataset, kernel ni package.

**S3** — Usuario: "solo agrega los nuevos polígonos que aparecieron con el umbral" (decisión previa aplicada a 273-300). Mantener el criterio de diferencia por IoU < 0.5 contra lo ya existente en CVAT antes de subir más polígonos.

**S4** — Restricción del proyecto: mapeo frame→imagen desde `data/raw/{finca}/meta.json`, nunca por aritmética. Verificar por hash antes de cualquier subida.

**S5** — Restricción del proyecto: CVAT solo se escribe con autorización explícita por lote. Preparación = datasets, kernels, corridas, packages y dry-run; sin PUT.

## Tasks

- T1 → S1,S4: bajar de CVAT (GET) las imágenes faltantes — Villas 0-199 (200) y Magdala 0-130 (131) — y completar `data/r1/images/`.
- T2 → S1,S2,S4: crear datasets de Kaggle con `frame_map.json` para Villas restante y Magdala restante.
- T3 → S2: kernels StarDist con `prob=0.30`, `nms=0.02`, export por frame. Correr en T4.
- T4 → S3: comparar salidas contra formas existentes en CVAT y preparar packages sólo con polígonos nuevos, por lotes.
- T5 → S5: dry-run de cada lote; esperar autorización antes de cualquier PUT.

## Log

L1 ( usuario, 2026-11-20): "prepara para los rangos sin inferencias"
L2 ( usuario, 2026-11-20): "c. no metas yulimar"

Estado inicial: 2311 frames con imagen local (Villas 449, Magdala 1110); 335 imágenes faltantes se bajan de CVAT. Sin Yulima.
