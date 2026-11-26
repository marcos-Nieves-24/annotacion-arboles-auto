# Plan — Búsqueda sistemática de alternativas para automatizar la anotación

Fecha: 2026-11-18. Alcance: **solo investigación y plan**. No entrena, no escribe en CVAT, no sube nada.

## 0. Contexto que condiciona la búsqueda

Problema real (de `docs/INFORME_2026-11-11.md`, `PLAN_234.md`, `PLAN_FRENTE_A.md`, `PLAN_TEXTURE_CLASSIFIER.md` y memoria de sesión):

- Copas de limón **muy juntas**: 68% de los frames de Villas tiene bboxes que se tocan; SAM2 automático y el modelo previo fusionan copas.
- Datos: **solo RGB**, tiles 512×512, GSD 3.58 / 5.25 / 3.84 cm/px (Villas / Magdala / Yulima). No hay multiespectral, LiDAR, CHM ni hiperespectral.
- Oro: 405 frames, YOLO11n-seg T1b en punto operativo conf 0.25 / NMS 0.35 / imgsz 640, **F1 0.728 en holdout, P 0.822 / R 0.653**. El cuello es **recall**, no precisión.
- Fallo de calidad identificado: en coronas densas que se tocan las máscaras salen rectangulares/recortadas (circularidad mediana 0.54–0.60).
- Ya descartado con números: SAM (F1 0.23), clasificador de textura v3 (rescate 100% pero FP 41%), transferencia de modelos externos (mAP 10–15%).
- Reglas duras: CVAT solo lectura; cero uploads sin autorización por lote; sin pseudo-labels ciegos (el usuario eligió oro limpio); preferir CPU/local.

Consecuencia: no buscamos "un modelo mejor" en abstracto. Buscamos **(a)** bajar el costo de anotación y **(b)** separar copas que se tocan, con RGB y sin sensor nuevo.

## 1. Preguntas de investigación

| ID | Pregunta | Métrica de respuesta |
|---|---|---|
| RQ1 | ¿Se puede igualar el F1 con menos anotación (puntos, cajas, 10–40% de frames, scribbles)? | F1 dev/holdout vs 0.728, y horas de anotación ahorradas |
| RQ2 | ¿Qué métodos separan copas que se tocan **sin** aprender de instancias (flujo, watershed, borde, geometría)? | F1 en frames densos + circularidad/solidity de máscara |
| RQ3 | ¿Qué variante de SAM/SAM2 con mejor prompt (punto, caja, elevación, texto) supera el F1 0.23 que obtuvimos? | F1 y FP en frames sin copas |
| RQ4 | ¿Cómo evitar la propagación de error en pseudo-labels (fine-tuning por etapas, confident learning)? | F1 con 20–40% de oro, sin caída de precisión |
| RQ5 | ¿Hay repos/datasets listos para correr en RGB local sin LiDAR? | Smoke test en 5 frames, CPU, sin dependencias bloqueantes |

## 2. Fuentes y protocolo de búsqueda

### 2.1 Fuentes
1. **OpenAlex** — metadatos bibliográficos y citas (`https://api.openalex.org/works`).
2. **Semantic Scholar Graph API** — abstracts, citas y PDF abierto.
3. **arXiv** — texto completo de preprints recientes (2025–2026).
4. **GitHub REST API** — repositorios, licencia, último push, viabilidad de código.
5. **Editoriales** (MDPI *Remote Sensing*, ISPRS Archives/Annals, IEEE GRSL/TGRS, *Computers and Electronics in Agriculture*, *Scientific Reports*) para el texto final.

### 2.2 Manejo de la API (lección de esta pasada)
- Los endpoints públicos de OpenAlex y Semantic Scholar devolvieron **HTTP 429** al lanzar varias consultas en paralelo desde una misma IP. No se usó ninguna credencial: no hace falta para el pool público.
- Protocolo: **máximo 1 request/s, lotes de ≤3, secuencial**, y cada respuesta se guarda en `data/literatura/raw/<fuente>_<slug>.json` con fecha. Nunca repetir una consulta ya cacheada.
- Si se quiere el "polite pool" de OpenAlex, el dueño del proyecto debe registrar su propio `mailto`; no inventar ni almacenar correos ni claves en el repo.
- GitHub: cabeceras `Accept: application/vnd.github+json` y `X-GitHub-Api-Version`; búsqueda limitada a ~10 req/min sin autenticar. Suficiente para esta revisión.

### 2.3 Matriz de queries (ejecutar en lotes)

**RQ1 — menos anotación**
- `semi-supervised individual tree crown segmentation UAV RGB`
- `weakly supervised tree crown delineation point annotation`
- `active learning annotation instance segmentation remote sensing`
- `sparse point annotations remote sensing image segmentation`
- `automatic annotation tree crowns foundation model prelabeling`

**RQ2 — copas que se tocan**
- `touching tree crowns instance segmentation orchard`
- `individual tree crown delineation dense canopy watershed boundary`
- `flow based grouping overlapping instances remote sensing`
- `tree crown separation distance transform edge segmentation`
- `overlapping orchards crown segmentation unsupervised`

**RQ3 — SAM / prompts**
- `segment anything model tree crown segmentation UAV RGB prompting`
- `SAM2 automatic mask generator tree crowns failure merging`
- `point prompt segment anything remote sensing tree crown`
- `learning to prompt remote sensing instance segmentation`

**RQ4 — pseudo-labels**
- `staged fine-tuning pseudo label noise tree crown segmentation`
- `confident learning noisy labels instance segmentation remote sensing`

**RQ5 — dominio y código**
- `citrus orchard tree detection UAV instance segmentation`
- `fruit orchard instance segmentation dataset`
- GitHub: `tree crown delineation`, `tree crown segmentation UAV`, `orchard instance segmentation`, `point supervised segment anything`, `segment anything remote sensing`, `citrus tree detection segmentation`

Filtros transversales: año ≥ 2019 (salvo papers fundacionales), acceso abierto preferido, y **excluir** explícitamente los que dependan de LiDAR/CHM/hiperespectral/multiespectral para funcionar.

## 3. Criterios de elegibilidad

**Incluir** si cumple todo:
- Entrada = imágenes ópticas RGB (o adaptable a RGB sin cambio de sensor).
- Ataca directamente RQ1–RQ4 (anotación o separación de instancias contiguas).
- Reporta métrica comparable (F1/IoU/mAP por instancia) en datos de dosel denso, plantación u huerto.
- Código abierto **o** descripción suficiente para reimplementar en <1 día.

**Excluir** si:
- Requiere LiDAR, CHM, nubes de puntos, hiperespectral o multiespectral como entrada obligatoria.
- Es solo clasificación por especie/estado vital (como el paper de Safonova ya revisado).
- Es segmentación semántica sin separación de instancias.
- Es un dataset sin método, salvo que sirva como preentrenamiento (marcar aparte).

## 4. Shortlist semilla (ya encontrada y verificada)

| # | Trabajo / repo | Fuente | RQ | Por qué importa |
|---|---|---|---|---|
| 1 | **FG-TreeSeg / ZS-TreeSeg** (Chen et al., IEEE GRSL 2026, arXiv 2602.00470) — `Pengyu-gis/ZS-TreeSeg` (MIT, push 2026-05) | arXiv + GitHub | RQ1, RQ2 | **Training-free**. Reformula la delimitación como flujo topológico y agrupa coronas que se tocan reutilizando un modelo semántico. mAP50 67.31 en BAMFORESTS vs 69.05 de Mask R-CNN supervisado; declara superar a TreePseCo separando coronas contiguas. README contiene el pipeline completo. Prioridad 1. |
| 2 | **Semi-supervised UAV ITC segmentation in heterogeneous planted forests** (2025) — `QingWWWW/treecrown-seg-finetune` (MIT, Python) | ScienceDirect + GitHub | RQ1, RQ4 | F1 0.826 / 0.836 / 0.711 con solo **40% del oro (0.6 h por stand)**; fine-tuning por etapas para frenar error de pseudo-labels; RGB supera a LiDAR. Casi nuestro escenario. Prioridad 2. |
| 3 | **PointSAM** (TGRS 2025) — `Lans1ng/PointSAM` (MIT, 77★, push 2026-11) | GitHub | RQ1, RQ3 | SAM con supervisión por **puntos** en teledetección. Un punto por copa es ~10× más barato que un polígono de 50 vértices. |
| 4 | **Manual Annotations Meet Fine-Tuned Foundation Models** (ISPRS Archives 2026) | OpenAlex | RQ3 | Comparación directa anotación manual vs modelo base afinado en segmentación de copas. Da el marco para decidir cuándo conviene anotar más. |
| 5 | **UAV RGB dataset + method for instance crown segmentation** (Sci Rep 2026) | Scientific Reports | RQ1, RQ3 | SAM con prompts cuidados **no superó** a Mask R-CNN; propone DeepMerge (similitud entre superpíxeles, sin etiquetas por píxel). Evidencia en contra de insistir con SAM. |
| 6 | **RSPrompter** (TGRS 2024, ~650 citas) | OpenAlex | RQ3 | Aprender prompts para segmentación de instancias en teledetección. Referente fuerte. |
| 7 | **ALPS** — `StriveZs/ALPS` (21★) | GitHub | RQ1 | Auto-etiquetado + preentrenamiento con SAM para segmentación RS. |
| 8 | **Tree Canopy Segmentation in Low-Data Regimes** (arXiv 2601.10931) | arXiv | RQ1 | Compara YOLOv11, Mask R-CNN, DeepLabv3, Swin-UNet y DINOv2 con **150 imágenes**. Nuestro régimen de datos. |
| 9 | **YOLOv5 + watershed** para coronas de picea (2025) | OpenAlex | RQ2 | Arquitectura parecida a la nuestra (YOLO + subdivisión por watershed). |
| 10 | **TreeCrownDelineation** — `AWF-GAUG/TreeCrownDelineation` (MIT, 118★, push 2026-01) | GitHub | RQ2, RQ5 | máscara + borde + distancia sobre imágenes ópticas; idea de splitting ya evaluada en el informe. |
| 11 | **detectree2** (274★) / **detectreeRGB** (41★) — `PatBall1/detectree2`, `shmh40/detectreeRGB` | GitHub | RQ5 | Mask R-CNN para ITCD; detectreeRGB es específicamente RGB. Bases comparables. |
| 12 | **MFO dataset** (CVPRW'25) — `tqosu/MFO` (MIT) | GitHub | RQ5 | Dataset de huertos frutales modernos con segmentación de instancias. Candidato a preentrenamiento. |
| 13 | **Overlapping orchards, satellite + unsupervised** (2021, PMC) | web | RQ2 | Segmenta filas y coronas en huertos de cítricos solapados sin aprendizaje supervisado. |
| 14 | **Csillik et al. 2018** — identificación de cítricos desde UAV con CNN; **Weinstein et al. 2019** — ITCD semi-supervisada en RGB | web | RQ1, RQ5 | Match de dominio (cítricos + RGB). |

Estado: semillas identificadas y metadatos de repo verificados vía API. **Aún no leído el texto completo de ninguna** — eso es el trabajo del plan.

## 5. Protocolo de screening

**Pasada 1 — Título/abstract (≈40 min).** Descartar por criterios de exclusión. Salida: tabla con `incluye/sí duda/no` y motivo de una línea.

**Pasada 2 — Método (≈2 h).** Para cada incluido, extraer en 5 campos fijos:
1. Entrada (¿RGB puro?).
2. ¿Separa instancias contiguas? ¿Cómo?
3. Supervisión requerida (píxel / punto / caja / ninguna) y cantidad.
4. Métrica reportada y en qué dataset.
5. Dependencias y hardware.

**Pasada 3 — Smoke test (≈1 h por repo, máximo 4 repos).** Clonar en `/tmp/user/1000/opencode/`, correr inferencia sobre **5 frames fijos** (2 densos, 2 con copas perdidas, 1 sin copas), sin tocar dev/holdout ni CVAT. Registrar: corre/no corre en CPU, tiempo/frame, dependencias faltantes, salida en formato máscara.

**Scorecard 0–5 por eje** (solo para los que llegan a pasada 3): impacto en recall, impacto en separación de coronas que se tocan, ahorro de anotación, madurez del código, compatibilidad RGB/CPU.

## 6. Gates de decisión

- **Gate 0 — viabilidad:** corre en RGB, CPU, 512×512, sin sensor nuevo. Si no, archivar con motivo.
- **Gate 1 — métrica:** evaluar con la **métrica unificada** (instance F1, IoU ≥ 0.5, one-to-one) en dev y holdout limpios, contra el baseline 0.728. Nunca usar mAP de Ultralytics como comparación.
- **Gate 2 — adopción:** aceptar solo si da **+0.02 F1** sin subir FP más de 2 pp, **o** el mismo F1 con ≥2× menos anotación. Si no, se descarta y se documenta con el número, igual que SAM y el clasificador de textura.
- **Orden de ejecución sugerido:** #1 FG-TreeSeg → #2 semi-supervisado por etapas → #3 PointSAM → #4 RSPrompter/ALPS. El orden prioriza "separar copas que se tocan" (nuestro fallo de calidad) antes que "abaratar anotación" (nuestro cuello de recall).

## 7. Entregables

1. `docs/ESTADO_DEL_ARTE.md` — tabla rankeada con las 14 semillas + lo que aparezca en las queries, con `incluye/descartado` y motivo.
2. `data/literatura/raw/*.json` — respuestas crudas de las APIs con fecha (evidencia, no se re-fetcha).
3. `data/literatura/fichas/*.md` — las 5 fichas de método de la Pasada 2.
4. `docs/DECISION_ALTERNATIVAS.md` — veredicto por candidato con el número del Gate 1 y el Gate 2.
5. Este plan queda **congelado**: cualquier búsqueda nueva se agrega como apéndice, no reescribe la matriz de queries.

## 8. Riesgos y mitigaciones

- **Sesgo de dominio:** casi toda la literatura es forestal, no citrícola. Mitigación: marcar dominio en cada ficha y desconfiar de números de bosque; el match real es huerto/plantación.
- **Papers 2026 muy recientes:** pocas citas y código sin madurar. Mitigación: el smoke test manda, no el paper.
- **Repetir el error de Rossi:** F1 1.0 por atajo de textura. Mitigación: medir siempre en dev/holdout limpios y con la métrica unificada.
- **Rate limits:** mitigados con caché y baja frecuencia (§2.2).
- **Alcance:** este plan **no autoriza** entrenar ni escribir en CVAT. Cada adoption requiere una decisión explícita después del Gate 2.

## 9. Log

- 2026-11-18: usuario pidió scrapear bibliografía y armar el plan de búsqueda con el contexto del proyecto.
- 2026-11-18: primera pasada real con OpenAlex + GitHub API (5 queries OA, 4 GH), dos 429 por límite de IP; complemento con websearch y Semantic Scholar (también 429). 14 semillas, 6 repos verificados por API (licencia, stars, push).
- 2026-11-18: Safonova 2021 (paper ITCD leído antes) queda descartado como alternativa: su foco es especie/estado vital y su punto débil es dosel denso, que es justo nuestro caso.
