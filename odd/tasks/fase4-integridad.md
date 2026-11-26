# Fase 4 — Oro extendido + YOLO+SAM

Objetivo: construir `gold_v2` con los 400 frames revisados y entrenar/integrar la arquitectura **YOLO + SAM** (detección de boxes + segmentación por prompt), reemplazando el pipeline R1b. Repo sin git (hash sha256 por entrega). Delivery: ask-on-risk. Runner: `.venv-verify/bin/python` (CPU) + Kaggle T4 (GPU, requiere autorización de subida).

## Specs

- S1: Usuario: "y si reentrenamos a R1b (Yolo) con dataset limpia. Y arma las demas capas de la arquitectura?" — alcance autorizado: reentrenamiento + arquitectura.
- S2: Usuario: "Solo oro 331, sin pseudo" → ampliado por S9. Principio: cero pseudo-labels en train.
- S3: Usuario (2026-11-18): "Yolo + SAM, no estamos hablando de opciones" — arquitectura elegida: YOLO detecta boxes → SAM segmenta cada copa con el box como prompt. Se descartan máscaras cuyo centro no cae en un box (Afsar 2024, Remote Sens 16(17):3207).
- S4: Prioridad: reducir over-split sin destruir copas reales. No usar conteo esperado para inventar/eliminar árboles. Casos ambiguos → revisión, nunca fusión/división automática sin evidencia.
- S5: No modificar gold_v1 ni el holdout. Todo va a `data/gold_v2/`.
- S6: Verificación rangos: Magdala 200-240 (41 frames, 736 shapes: 582 semi-auto + 154 auto), Villas 200-232 (33 frames, 339 shapes: 259 semi-auto + 80 auto, vacíos 202/203/204/211/227). Usuario confirmó: "yo las revisé" → se incluyen TODAS; `source` no filtra. Auditoría de duplicados IoU>0.9 obligatoria en T0.
- S7: gold_v2 = 400 frames, ~4087 instancias antes de GSD. Protocolo v1: Douglas-Peucker eps 1.5px, GSD Villas 882-39224 / Magdala 410-18240, iscrowd si ≥3 vértices en borde, cuarentena con motivo, splits A/B por finca, holdout v1 congelado, dev 10% por finca para umbrales (nunca para reportar).
- S8: Evidencia previa: R1b conf 0.5 → F1 0.756 en 58 frames oro; techo barrido 0.762; RF actual AP 0.9997 engañoso (separa por área); 196 FP de R1b en boxes. Por eso YOLO+SAM usa boxes de un modelo reentrenado limpio y filtra máscaras por centro-en-box.
- S9: DeepForest evaluado y descartado (Fase 1: box mAP50 10.2%, MAE 24.3; solo cajas). No se instala.
- S10: Métricas: matching uno-a-uno IoU≥0.5; instance P/R/F1, over-split rate, under-split rate, duplicate rate, FP rate, count MAE; desglose por finca; recall ≥ baseline R1b − 2pp. Holdout 20 tocado una vez al final.

## Tasks

- T0 (S6,S7) — inline — `scripts/build_gold_v2.py`: GET a CVAT (solo lectura) para jobs 4307818 y 4307835 + oro v1, incluir TODAS las shapes de los rangos, auditoría de duplicados geométricos (IoU>0.9), convertir con protocolo v1, excluir holdout, splits A/B, dev 10%, `quarantine.json`, `conversion_report.json`. Salida: `data/gold_v2/`. Estado: pendiente.
- T1 (S3,S8) — inline — adaptar kernel `fase4-clean` (ya generado, sha `76d1b553...`) a gold_v2: YOLO11n-seg reentrenado sin pseudo, 150 ep, imgsz 640. Salida: pesos `best.pt` por split. Subida a Kaggle requiere autorización.
- T2 (S3) — inline — `scripts/yolo_sam_pipeline.py`: cargar pesos YOLO de T1, `predict(conf, iou)` → boxes; por cada box, prompt a SAM (box + puntos opcionales) → máscara; descartar máscaras cuyo centro ∉ box; NMS; output COCO/YOLO-seg + overlays + reporte. Instalar SAM (`segment-anything`) en `.venv-verify`. Estado: pendiente.
- T3 (S10) — inline — `scripts/eval_yolo_sam.py`: benchmark T2 contra gold_v2 (dev para umbrales, holdout 20 al final), comparar vs baseline R1b, reporte por finca. Estado: pendiente.

## Log

- L1: Usuario (2026-11-17): "y si reentrenamos a R1b (Yolo) con dataset limpia. Y arma las demas capas de la arquitectura?"
- L2: Usuario: "Solo oro 331, sin pseudo".
- L3: GPU local `cuda: False`; Kaggle CLI 2.2.4 con credenciales; repo sin git.
- L4: T1 kernel `fase4-clean` generado (6 celdas, sha `76d1b553fb6636f3`). No subido.
- L5: DeepForest evaluado, descartado.
- L6: RF reevaluado (`rf_reeval.py`): AP 0.9997 engañoso; no adoptado.
- L7: Usuario (2026-11-18): "podemos tomar 200-240 de Magdala y 200-232 para aumentar. Ya están revisados. arma el plan completo de la arquitectura."
- L8: Usuario: "yo las revisé. quiero que planifiques la nueva arquitectura" → toutes les shapes incluidas.
- L9: Usuario: "para que T2-T5? si vamos a instalar una nueva arquitectura" → se eliminan módulos propios redundantes.
- L10: Usuario (2026-11-18): "Yolo + SAM, no estamos hablando de opciones" → arquitectura fijada. Plan reescrito a T0-T3.
- L11: Siguiente: T0 construir gold_v2. Sin autorización de Kaggle todavía.
