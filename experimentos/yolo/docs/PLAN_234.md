# Plan 2-3-4 — Mejora de output sin re-entrenar (todo local, CPU)

Base: pesos r1 (`runs1b/r1/best.pt`), oro 331, muestra 90, scripts existentes.
Prohibido: GPU/Kaggle/CVAT/uploads. Todo corre en `.venv-verify`.

## Punto 2 — Postproceso v1 (benchmark y congelado)

**Objetivo:** un solo postproceso estándar con número ganador.

**Paso 2.1 — Instalar `supervision` en el venv** (~5 min)
```
.venv-verify/bin/pip install -q supervision
```

**Paso 2.2 — Benchmark NMM vs count-merge** (~20 min CPU)
Script `scripts/bench_nmm.py` (a crear, 60 líneas):
- Entrada: muestra 90 + pesos r1.
- Rama A (nuestro): conf 0.5, NMS 0.35, área GSD, count-merge.
- Rama B (externo): `sv.Detections` desde cajas+máscaras r1 →
  `with_nmm(threshold=T)` para T en {0.05, 0.1, 0.15, 0.2}.
- Métrica: P/R/F1 máscara vs referencia, **solo en frames con oro
  limpio** (Villas 0-199 y Magdala 0-130 dentro de la muestra; el resto se
  reporta aparte como acuerdo-vs-auto, no como calidad).
- Criterio: gana mayor F1 con recall ≥ mejor-recall − 0.02.

**Paso 2.3 — Congelar `scripts/postprocess_v1.py`**
El ganador empaquetado como función única
`corregir(frame) -> labels + overlay + conteo`, con parámetros
(conf 0.5, NMS 0.35, área GSD por finca, umbral NMM o count-merge ganador).
Sin re-entrenar, sin GPU.

**Riesgo:** NMM une de más en setos (une por IoU bajo). Mitigación: el
benchmark lo detecta (cae precisión en Yulima denso) y se limita su umbral
o se restringe a Villas/Magdala.

## Punto 3 — Estimador de conteo operativo

**Objetivo:** cada frame sale con conteo esperado, final y bandera QC.

**Ya existe:** `esperado = área_dosel_unión / copa_mediana_finca`
(Villas 9798, Magdala 6636, Yulima 4057 px²).

**Paso 3.1 — Reporte por frame** (sale del mismo script del punto 2):
`n_inicial, n_esperado, n_final, bandera ∈ {ok, revisar_fusion, revisar_falta}`.
Reglas: si final > 1.3×esperado → `revisar_fusion`; si < 0.7× → `revisar_falta`.

**Paso 3.2 — Corrección automática acotada:** solo fusionar (nunca inventar
detecciones). Límite: como máximo se fusiona hasta `1.1×esperado + 1`.
Lo que quede fuera de banda va a lista de revisión, no se toca.

**Criterio:** en muestra con oro, el conteo final debe acercarse al conteo
real (MAE por frame) sin bajar el F1 del punto 2.

## Punto 4 — Cierre de herramientas externas

**Estado:** NMM (`supervision`, MIT) es la única pieza aplicable RGB-only.
CrownSeg (pide hiperespectral/LiDAR, Ubuntu 18.04) y pycrown (pide CHM
LiDAR, archivado 2021) descartados con motivo en informe.

**Paso 4.1:** el benchmark del punto 2 ES el cierre: si NMM gana, se adopta
y se documenta versión; si pierde contra count-merge, se archiva con el
número que lo prueba. No hay más búsqueda abierta después de esto.

## Entregables (verificables por vos)

1. `data/mejora_v1/{labels,overlays,reporte_conteo.json}` en la muestra 90.
2. Tabla antes/después con F1 y MAE de conteo.
3. `scripts/postprocess_v1.py` reutilizable.
4. Esta decisión documentada (quién ganó y por cuánto).

## Tiempos y costo

~1h CPU total, 0 GPU, 0 subidas, 0 toques a CVAT. Si algo falla, todo es
re-ejecutable localmente.
