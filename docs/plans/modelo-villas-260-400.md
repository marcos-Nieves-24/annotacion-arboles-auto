# Plan — Aplicar el modelo a Las Villas, frames 260-400

> Plan de ejecución (no implementación). Estado: **borrador con decisiones a confirmar**.
> Alcance: inferencia local sobre un rango nuevo de frames; **cero escrituras en CVAT**.

## Outcome

Aplicar el modelo entrenado y canónico a los frames **260-400 inclusive** de la finca
`2475966 Las_Villas` (141 frames, `Limon_LasVillas_00770.png` … `Limon_LasVillas_00968.png`)
para producir **pre-etiquetas de segmentación**, **overlays para revisión humana** y un
**reporte de QC**, sin tocar el oro, el holdout ni CVAT.

Antecedente: ya se prepararon payloads R1 para Villas **220-230** (11 payloads, 73 polys)
y **231-250** (20 payloads, 140 polys), job CVAT 4307835 / task 2475966. El formato
concreto es `{"shapes":[{"type":"polygon","label_id":8155070,"frame":<índice>,"points":[...]}]}`
(ver `/tmp/user/1000/opencode/villas_Limon_LasVillas_00710_payload.json`). Este rango
continúa esa secuencia: 260-400.

Criterios de aceptación:

1. 141/141 frames procesados (o listado explícito de faltantes en `missing_frames.json`).
2. Salida en `data/villas_260_400/`: `labels/*.txt` (YOLO-seg), `overlays/*.png`,
   `contact_sheet.jpg`, `report.json` y, si se confirma D3, `payloads/villas_<name>_payload.json`
   con el mismo formato que los payloads 220-250.
3. Cada máscara convertida a polígono con el mismo protocolo v1 del proyecto
   (Douglas-Peucker eps 1.5 px, filtro de área GSD Villas 882–39224 px²).
4. `report.json` incluye por frame: `n_det`, `n_esperado`, `n_final`, `flag`
   (`ok` / `revisar_fusion` / `revisar_falta` / `over_cap`), y confidencia usada.
5. Verificación visual de una muestra estratificada (rálido/medio/denso) antes de
   cualquier uso posterior. "Hecho" = artefactos en disco + muestra revisada, **no**
   "subido a CVAT".

## Evidencia de partida (verificada en este checkout)

| Hecho | Evidencia |
|---|---|
| Modelo canónico = `model_final.pt`, YOLO11n-seg (T1b), 405 frames oro, sin early stopping | `scripts/build_kernel_fase4_t1b.py:5,101-103`; `data/fase4/eval_unified_t1b.json` |
| Peso disponible localmente (6 MB), CPU funcional | `/tmp/user/1000/opencode/wm/model_final.pt` |
| Receta de inferencia ya usada: `YOLO(W).predict(..., imgsz=640, iou=0.35)` + máscaras→contornos | `scripts/make_overlays_t1b.py:35-65`, `scripts/make_overlays_model_only.py:33-57` |
| Frames 260-400 existen localmente | `data/raw/villas/meta.json` (f[260]=`…00770.png`, f[400]=`…00968.png`); `data/r1/images/villas/` tiene 489 frames (200-689) |
| Punto operativo medido en dev (38 frames, imgsz 640): conf 0.25 → P .783/R .651/F1 .715; conf 0.35 → P .837/R .625/F1 .716; conf 0.5 → P .870/R .558/F1 .680 | `data/fase4/sweep_conf_imgsz.json` |
| `conf_sweep` en 58 frames da meseta F1 .756 en conf 0.5-0.65 | `data/fase4/conf_sweep.json` (set distinto: reconciliar visualmente, no promediar) |
| Filtro de área y merge guiado por conteo ya implementados para Villas | `scripts/ronda2.py:15-16,21-44` |
| `segment_anything` y `ultralytics` instalados; `supervision` **no** instalado | `.venv-verify/bin/python` |
| Regla del proyecto: CVAT solo lectura, cero escrituras | `odd/tasks/fase4-integridad.md:3`; `INFORME_2026-11-11.md:4` |
| Pipeline YOLO+SAM (T2) **no construido** (`scripts/yolo_sam_pipeline.py` no existe) | `odd/tasks/fase4-integridad.md:22` |
| Payloads R1 ya preparados para 220-230 y 231-250 (job 4307835); formato `shapes[].{type,label_id,frame,points}` | memorias #1566/#1583; `/tmp/user/1000/opencode/villas_Limon_LasVillas_00710_payload.json` |
| La numeración de nombres **no es contigua** (salto `00733→00735`, `00735→00753`) → mapear por índice de `meta.json`, nunca por sufijo numérico | memoria #1583; `data/raw/villas/meta.json` |

## Approach

Slice mínimo que cruza el límite real: **un script nuevo de inferencia + artefactos**, reutilizando
el peso y la receta existentes. Sin re-entrenar, sin GPU, sin Kaggle, sin SAM, sin CVAT.

1. **Paso 0 — fijar el peso.** Copiar `/tmp/user/1000/opencode/wm/model_final.pt` a
   `data/weights/model_final.pt` (el original vive en `/tmp`, es efímero). Registrar sha256.
2. **Paso 1 — script `scripts/run_model_villas_260_400.py`:**
   - Leer `data/raw/villas/meta.json`, tomar `names[260:401]` (141 frames). **Mapear por
     índice de lista, no por sufijo numérico** (hay saltos `00733→00735→00753`). Verificar
     existencia en `data/r1/images/villas/`; escribir `missing_frames.json` y continuar.
   - Cargar modelo desde `data/weights/model_final.pt`.
   - Predecir en batches de 25-40: `imgsz=640, conf=0.25, iou=0.35, verbose=False`.
     Imprimir progreso por batch y reporte parcial siempre (convención del repo).
   - Por detección: máscara `>0.5` → resize 512 si hace falta → `findContours` →
     `approxPolyDP` → área GSD; descartar fuera de banda; normalizar a 512 para YOLO-seg.
   - Escribir `labels/<stem>.txt` (`0 x1 y1 x2 y2 …`), `overlays/<name>.png`
     (máscara + box + conf), y `contact_sheet.jpg` (5 col, estratificado).
   - Reporte por frame con `n_det`, `n_esperado` (área_dosel_unión/9798),
     `n_final`, `flag`; reglas: `>1.3×esperado → revisar_fusion`,
     `<0.7× → revisar_falta`, `>40 det → over_cap`.
3. **Paso 2 — postproceso conservador.** Aplicar **solo** el filtro de área GSD y
   las banderas de conteo. **No** fusionar/dividir automáticamente: la spec S4 de fase 4
   prohíbe inventar o eliminar árboles sin evidencia. `count_guided` de `ronda2.py`
   queda como opción desactivada (`--count-guided`), no default.
4. **Paso 3 — payloads (según D3).** Si se confirma la continuidad del flujo 220-250,
   emitir un payload por frame con `frame` = índice 260-400, `label_id` = 8155070 y
   `points` = vértices del polígono en píxeles (mismo esquema que los 31 payloads que
   siguen en `/tmp`). Escribirlos en `data/villas_260_400/payloads/`. **Generar el archivo
   no es subirlo**: el PATCH a CVAT queda fuera de este plan salvo autorización explícita.
5. **Paso 4 — revisión.** Contact sheet + overlays para revisión humana. La corrección
   en CVAT, si se decide, es un paso **separado y autorizado explícitamente**.

## Decisions

| # | Decisión | Elección propuesta | Evidencia | Qué la cambiaría |
|---|---|---|---|---|
| D1 | ¿Qué es "el modelo"? | `model_final.pt` (YOLO11n-seg T1b), el artefacto entrenado que existe | T1b lo produce; T2 (YOLO+SAM) no está construido ni validado | Que quieras específicamente la arquitectura YOLO+SAM → hay que armar T2 primero (otro plan) |
| D2 | Confidencia | 0.25 (recall-biased para pre-etiquetado) con `imgsz=640, iou=0.35` | dev sweep: 0.25 maximiza recall y F1 ≈ 0.35; meseta distinta en `conf_sweep` | Que haya muchos FP en la muestra visual → subir a 0.35/0.5 |
| D3 | Destino de salida | Local + **payloads con formato CVAT** (continúa 220-250), sin PATCH | payloads 220-250 existen en `/tmp`; regla de solo lectura | Que quieras solo labels/overlays, o autorización explícita de PATCH a CVAT |
| D4 | Merge guiado por conteo | Desactivado por default | S4: sin borrado automático sin evidencia | Evidencia visual de over-split sistemático |
| D5 | Peso en `/tmp` | Copiar al repo antes de correr | `/tmp` es efímero; el job se pierde | — |

Supuestos **no verificados** (confirmar con el humano):
- "260-400" son índices del `meta.json` (0-based), no números de frame `00770`…`00968` ni ids de CVAT.
- "Usar el modelo" = pre-etiquetar para revisión (con payloads locales, sin PATCH); no implica subir a CVAT ni reentrenar.
- Las imágenes de `data/r1/images/villas/` son las correctas para esos índices (mismo `meta.json`); los saltos de numeración (`00733→00735→00753`) se respetan mapeando por índice.

## Boundaries

Fuera de alcance: tocar `gold_v1`/`gold_v2`, `holdout`, splits, pesos de entrenamiento;
reentrenar; instalar `supervision`; construir el pipeline SAM; escribir en CVAT; procesar
frames fuera de 260-400. Si aparece un frame faltante, se reporta, no se descarga en este plan.

## Verification

| Criterio | Cómo se verifica | Fallo esperado |
|---|---|---|
| 141 frames resueltos | `ls data/villas_260_400/labels/*.txt \| wc -l` y `missing_frames.json` | Faltantes listados, pipeline continúa |
| Salida válida | `head` de un `.txt`: `0` + pares `x y` en [0,1] | Archivo vacío si 0 detecciones → se registra, no se omite |
| Protocolo v1 aplicado | comparar un polígono con el overlay; área dentro de 882–39224 | Máscara fuera de banda → descartada y contada en reporte |
| Bandera de conteo | `report.json`: recomendar revisión donde `n_final > 1.3× esperado` | `over_cap` se marca, nunca se descarta silenciosamente |
| Determinismo | re-correr 2 frames, comparar `.txt` byte a byte | Distinto → revisar resize/contornos |
| Payloads coherentes | 141 JSON con `shapes[].frame` ∈ [260,400], `label_id`=8155070 y `points` pares; comparar esquema con el payload 220 | `frame` fuera de rango o `points` impar → error |
| Sin CVAT writes | revisar que el script no contenga `PATCH`/`action=create` | Cualquier aparición bloquea la ejecución |
| Peso reproducible | `sha256sum data/weights/model_final.pt` vs original | Distinto → recopiar |

## Docs

Ningún doc "corriente" queda invalidez por este plan: es un artefacto nuevo en
`docs/plans/`. Si se adopta un punto operativo de conf distinto al de los scripts
existentes (`0.5` en `make_overlays_t1b.py`), registrar el número y la evidencia en este
mismo plan y en el log de `odd/tasks/fase4-integridad.md` (sin reescribir specs cerradas).

## Recovery

Todo es regenerable y local: borrar `data/villas_260_400/` y re-corrrer con otro `conf`
no afecta oro, holdout ni CVAT. Si el peso copiado se corrompe, recopiar de
`/tmp/user/1000/opencode/wm/model_final.pt`; si `/tmp` ya no existe, hace falta
recuperarlo del output del kernel de Kaggle (requiere autorización de subida/descarga).
No hay migraciones ni escrituras irreversibles en este plan.
