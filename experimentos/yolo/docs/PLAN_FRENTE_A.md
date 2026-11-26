# Frente A — Reentrenamiento con hard examples (copas no marcadas)

Objetivo: recuperar las copas reales que YOLO T1b no detecta (problema A), sin quemar dev/holdout y sin reintroducir pseudo-labels.

## 1. Hallazgo de ubicación de los frames auditados

Los 8 frames que el usuario confirmó visualmente ("copas reales claras no detectadas") están así en gold_v2:

| Frame | Copas (usuario) | Set actual | Acción |
|---|---:|---|---|
| 00509 | ~4 | DEV | ⚠️ no tocar train |
| 00196 | ~2 | DEV | ⚠️ no tocar train |
| 00499 | ~12 | DEV | ⚠️ no tocar train |
| 00563 | ~5 | DEV | ⚠️ no tocar train |
| 00519 | ~2 | DEV | ⚠️ no tocar train |
| 00626 | ~2 | TRAIN | ✅ se puede sobre-muestrear |
| 00866 | ~13 | HOLDOUT | ❌ nunca tocar |
| 00946 | ~4 | HOLDOUT | ❌ nunca tocar |

Consecuencia: **solo 1 de los 8 auditados está en train**. Los 5 de dev no pueden entrar a train sin destruir la validación, y los 2 de holdout son sagrados.

## 2. Estrategia correcta: hard-example mining sobre TRAIN, no sobre los auditados

En vez de usar los frames auditados (que están en dev/holdout), aplicar la **misma condición** a los frames de TRAIN:

1. Correr YOLO T1b (conf 0.25) sobre los **347 frames de train**.
2. Identificar frames de train con **recall bajo**: frames donde el conteo YOLO es mucho menor que el conteo de copas gold (o donde hay copas gold sin ninguna detección que las cubra).
3. Esos frames de train son los "hard examples" — sobre-muestrearlos ×3 en el `train.txt`.

Esto ataca la causa (copas chicas/duras) sin tocar dev ni holdout.

## 3. Hyperparámetros (mismo kernel T1b, con los fixes aprendidos)

| Parámetro | Valor | Cambio vs T1b |
|---|---|---|
| Arquitectura | yolo11n-seg.pt desde pretrained | igual |
| epochs | 150 | ↑ de 120 |
| imgsz | 640 | igual |
| batch | -1 | igual |
| device | 0 (T4) | igual |
| workers | 2 | igual |
| patience | 30 | ↑ de 0 (evita peso corrupto) |
| exist_ok | False | igual (directorio limpio) |
| optimizer / lr | auto (AdamW) / 0.01 | igual |
| mosaic / flip / hsv | 1.0 / 0.5 / 0.015 | igual |
| copy_paste / mixup | 0 / 0 | igual (cóctel colapsó en 2d) |
| seed | 0 | igual |
| **hard frames** | ×3 en train.txt | **nuevo** |
| **peso guardado** | recargar best.pt antes de val | **nuevo** (fix bug) |

## 4. Qué NO se hace

* No se meten frames de dev ni holdout en train (regla rota si se hace).
* No se usan pseudo-labels (usuario eligió oro limpio).
* No se baja conf (ya medido: trae maleza).
* No se usa SAM (F1 0.23, descartado).
* No se usa el clasificador de textura (41% FP, descartado).

## 5. Métricas y criterio

Misma métrica unificada: instance F1 con IoU≥0.5, one-to-one, en dev y holdout.

| Métrica | Baseline actual | Objetivo |
|---|---|---|
| F1 holdout | 0.647 (conf 0.5) / 0.728 (conf 0.25) | ≥ baseline + 0.02 |
| Recall holdout | 0.518 / 0.653 | ↑ sin bajar precisión >2pp |
| F1 dev | 0.680 / 0.715 | consistente con holdout |

**Criterio de éxito**: subir recall en el holdout sin que la precisión caiga >2pp. Si no, se descarta y se acepta 0.728 como techo.

## 6. Orden

1. Script `select_hard_train.py`: correr T1b sobre 347 frames de train, marcar los de bajo recall.
2. Generar `train.txt` con esos frames ×3.
3. Adaptar kernel T1b → T1c con los hypers de arriba.
4. Entrenar en Kaggle T4.
5. Evaluar en dev/holdout limpios con métrica unificada.
6. Decidir: adoptar o descartar.

## Log
- Usuario: "como seria el frente A" (2026-11-18).
- Hallazgo: 5 auditados en DEV, 2 en HOLDOUT, solo 1 en TRAIN → el plan usa hard-mining sobre TRAIN, no los auditados directamente.
