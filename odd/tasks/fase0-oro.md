# Fase 0 — Fijar el oro (gold_v1)

Objetivo: congelar 331 frames revisados como GT v1.0 para entrenamiento.
Alcance: solo lectura en CVAT + trabajo local. NO subir nada a CVAT.

## Specs
- S1: Villas frames 0-199 (200 frames, Limon_LasVillas_00126-00670) + Magdala 0-130 (131 frames, Limon_Magdala_00222-01088). Verbatim usuario: "villa: del frame 0-199 está revisados / magdala: del frame 0-130".
- S2: GSD asumido plan de vuelo: Villas 3.58, Magdala 5.25 cm/px (notebook automatic-labeling.ipynb, celda FARMS).
- S3: Filtros métricos 1.2-8.0m diámetro → Villas 882-39224px, Magdala 410-18240px. Fuera de rango = cuarentena, no train.
- S4: Split por finca (train Villas / val Magdala + inversión), nada de random global.
- S5: Protocolo 1 página: corte por concavidad, borde truncado = iscrowd, grupo inseparable aparte, vacío verificado.

## Tasks
- T1: Descargar annotations.json + frames de los rangos oro (solo GET). Estado: hecho. 331/331 PNG 512x512, 0 faltantes, 181MB.
- T2: Conversión COCO + YOLO-seg con simplificación 30-50 vértices. Estado: hecho. 331 imgs, 3548 kept, 217 cuarentena, 636 iscrowd, 2912 YOLO. Vértices mediana 23→17.
- T3: splits.json + protocolo + reporte. Estado: hecho (splits por finca A/B, PROTOCOLO.md v1.0).

## Log
- L1: Usuario: "empieza por fase 0" (2026-11-11). Autoriza trabajo local, CVAT read-only.
- L2: Conversión con DP epsilon 1.5px. iscrowd >=3 vértices de borde (1-2 = recorte leve, entrena). Área por GSD: 217 cuarentena (mayoría splinters <mínimo + mega-fusiones). Revisado: vértices oro más limpios que tarea completa (mediana 23 vs 50).
- L3: Dataset Kaggle privado `marcosalvarez234/limon-gold-v1` v2 live (status ready): coco, yolo implícito, splits, holdout, quarantine, protocolo + images.zip 180MB. Runbook en docs/kaggle-baseline-runbook.md.
- L4: 300 overlays Villas (frames 0-299) en `data/gold_v1/overlays_villas/` (158MB, todo local por pedido: verde=oro, azul=auto, rojo=modelo B). Spot-check: modelo calca el auto en aislados; fragmenta en denso bajo.
- L5: Fase 2c cerrada: m-seg A 83.3/B 82.2 vs nano 83.1/86.4 → se queda el nano B como operativo. Pesos m en `data/kaggle-out-fase2c/runs-m/*/weights/best.pt`.
