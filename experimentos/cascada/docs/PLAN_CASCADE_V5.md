# Plan — cascade-v5: resolución de intersecciones por prioridad de confianza

Fecha: 2026-11-19. Estado: **plan** (no implementado). Alcance: solo postproceso sobre v4; no se reentrena ni se tocan los pesos v1 (`ec2bc282e40bd94a`).

## 1. Motivación

v4 detecta bien las copas (F1 dev 0.933 / holdout 0.900) pero deja máscaras que se pisan: en los 10 frames de prueba 261–270 hay 39 pares con IoU > 0.05 y 13 con IoU > 0.15.

Las alternativas probadas fallaron:

| Intentos | Resultado |
|---|---|
| Voronoi / watershed sobre distancia (opt2) | IoU 0 pero recorta el 36% del área; forma angulosa |
| watershed sobre heatmap (opt1) | lo mismo |
| selección mínima de solape entre las 3 de SAM (opt3) | reduce solapes pero pierde 7 copas |

La alternativa de este plan viene de un trabajo ISPRS 2026: **ordenar máscaras por confianza y pintar en orden**, de modo que una máscara solo escriba píxeles no reclamados por otra de mayor confianza.

## 2. Evidencia local (ya medida, sin escribir en Kaggle)

`scripts/preview_priority_261_270.py` corrió la cascada v4 sobre los 10 frames y aplicó priority painting en memoria:

| | Pares IoU>0.05 | IoU>0.15 | Max IoU | Area ratio | Circ |
|---|---:|---:|---:|---:|---|
| v4 baseline | 39 | 13 | 0.239 | 1.04 | 0.57 |
| prioridad score del modelo | 0 | 0 | 0.000 | 0.92 | 0.55 |
| **prioridad score de SAM** | **0** | **0** | **0.000** | **0.99** | **0.57** |

El score propio de SAM es mejor que el del modelo de centroides: mantiene área y forma casi idénticas. Overlays en `data/centroides/priority_261_270/`.

## 3. Cambios exactos en el kernel

Archivo nuevo: `kernels/centroides_cascade_v5/kernel_source.py`, copia de `kernels/centroides_cascade_v4/kernel_source.py`.

### 3.1 Capturar el score de SAM

Hoy el bucle de inferencia es:

```python
ms, sc, _ = predictor.predict(box=box, multimask_output=True)
masks.append(ms[int(np.argmax(sc))])
```

Cambiar a:

```python
ms, sc, _ = predictor.predict(box=box, multimask_output=True)
k = int(np.argmax(sc))
masks.append(ms[k])
sam_scores.append(float(sc[k]))
```

Inicializar `sam_scores = []` antes del bucle.

### 3.2 Emparejar scores con las máscaras aceptadas

Después de `validate`, armar un mapa por centro (clave `(round(cx,1), round(cy,1))`) y construir la lista de items con su score, ordenada descendente.

### 3.3 Priority painting

Insertar **después** de `annotate_confidence` (para no perder la información de triaje de v4) y **antes** de `match_masks`/dibujo de overlays:

```python
def priority_paint(items):
    claimed = np.zeros((512, 512), bool)
    out = []
    for it in sorted(items, key=lambda x: -x["sam_score"]):
        m = it["mask"]
        out.append((it["prop"], m & ~claimed))
        claimed |= m
    return out
```

`kept` pasa a ser la lista pintada. `rejected` y `ambiguas` se calculan sobre las máscaras originales y se conservan en el reporte.

### 3.4 Reporte

Agregar a cada split:

- `intersecciones_despues`: nº de pares con IoU > 0.05 y > 0.15 sobre las máscaras pintadas.
- `pixeles_cedidos_por_frame`: total de píxeles que cada máscara perdió.
- `area_ratio_med` y `circularidad_med` antes/después.

## 4. Métricas y gates

| Métrica | Línea base v4 | Criterio |
|---|---|---|
| F1 dev / holdout | 0.9332 / 0.8998 | no bajar más de 0.002 |
| Precisión holdout | 0.9154 | no bajar más de 0.002 |
| Pares con IoU > 0.15 (dev+holdout) | 13 (de 58 frames) | **0** |
| Area ratio mediano | ~1.0 | entre 0.85 y 1.15 |
| Pares ambiguos reportados | 11 | se conservan (info de revisión) |

Si F1 cae más allá del criterio, se descarta y se mantiene v4.

## 5. Orden de ejecución

1. Crear `kernels/centroides_cascade_v5/` desde v4 con los tres cambios.
2. Verificar con el test local de 10 frames (`scripts/preview_priority_261_270.py` ya valida la lógica).
3. Build + `kaggle kernels push` con id `marcosalvarez234/limon-centroides-cascade-v5`, `kernel-source: limon-centroides-v1`.
4. Bajar reporte y review; comparar con v4.
5. Si pasa los gates, actualizar `docs/CALIBRACION_VALIDADOR.md` y el ODD; si no, documentar el descarte con números.

## 6. Riesgos

- El paint depende del orden por score: si SAM puntúa mal una copa grande, la copa pequeña vecina se queda con sus píxeles. Mitigación: el área ratio se reporta por frame; si cae, probar score compuesto `sam_score × area_ratio`.
- Los pares marcados como ambiguos en v4 ahora tienen IoU 0, así que el detector dejaría de marcarlos si se corriera después. Por eso se conserva el `ambiguas` calculado antes del paint.
- No resuelve el caso de una copa literalmente encima de otra: requiere altura o flujo, no postproceso 2D.

## 7. Log

- 2026-11-19: evidencia local en 10 frames; decisión de usar score de SAM como prioridad; plan escrito sin tocar kernels.
