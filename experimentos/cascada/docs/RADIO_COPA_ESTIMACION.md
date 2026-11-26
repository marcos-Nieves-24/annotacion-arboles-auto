# Estimación del radio de la copa — análisis y opciones

Fecha: 2026-11-18. Alcance: **análisis y recomendación**. No se reentrenó nada; la v1 sigue congelada (`docs/MODELO_CENTROIDES_V1.md`).

## 1. El problema

La cascada usa un radio fijo por finca (Villas 58.5 px, Magdala 45.4 px, Yulima 35.9 px) para:

1. armar el box prompt de SAM (`±1.15·r`),
2. la banda de área del validador (`[0.35, 2.0]·π·r²`),
3. la distancia del NMS entre centros (`0.7·(r_i+r_j)`).

Cuando el radio es menor que la copa → el círculo no cubre todo. Cuando es mayor → cubre suelo o vecinos. En los 9 frames auditados, **65 de 84 rechazos fueron por `area`**.

## 2. Hallazgos medidos

Sobre las 4602 copas del oro y 80 frames de muestra:

| Medición | Valor |
|---|---|
| Radio equivalente `√(area/π)` — mediana / p10 / p90 | 51.4 / 25.1 / 70.3 px |
| `r_enclose / r_eq` (radio que realmente encierra el polígono) | mediana 1.32, p90 1.93, máx 2.98 |
| Un círculo de `1.15·r_eq` cubre la copa completa | **solo 10.2% de los casos** |
| `distancia al vecino / 2` contra `r_eq` | error relativo mediano 21.5%, ratio mediano 1.00 |

Sobre 40 frames con el heatmap predicho del modelo v1:

| Estimador desde el heatmap | Correlación (Spearman) con `r_eq` |
|---|---|
| tamaño del blob a umbral 0.5 | 0.699 |
| tamaño del blob a umbral 0.7 | 0.722 |
| tamaño del blob a umbral 0.9 | **0.772** |

El factor teórico del target (`r ≈ 13.6·radio_del_núcleo_en_heatmap`) **no aplica al heatmap predicho**, que es más ancho. El factor empírico calibrado es:

```
r_est ≈ 2.17 · √(área_{heat>0.5}/π) · 4      [píxeles de imagen]
```

Con ese factor: error relativo mediano **15.8%** (p25 7.8%, p75 26.8%), 60% de las copas con error < 20%, ratio `r_est/r_real` p25 0.88 / mediana 1.00 / p75 1.22.

## 3. Opciones

### A. Radio por copa desde el tamaño del blob del heatmap ✅ recomendada

- **Qué**: para cada pico, contar píxeles del heatmap > 0.5 en una ventana ±10 celdas y aplicar el factor 2.17·4.
- **A favor**: no cuesta inferencia extra (el heatmap ya existe), no requiere reentrenar, error mediano 15.8%, correlación 0.70–0.77.
- **En contra**: cuando dos copas se tocan, sus blobs se unen y sobreestima; hay que limitar el resultado a `[0.6, 1.6]·mediana_de_la_finca`.
- **Uso**: reemplazar el radio fijo en el box prompt, la banda de área y el NMS.

### B. Distancia entre centroides vecinos

- **Qué**: `r_i ≈ distancia_al_centroide_más_cercano / 2`.
- **A favor**: gratis con las propuestas ya calculadas; error mediano 21.5%; funciona bien en huertos en cuadrícula.
- **En contra**: falla en árboles aislados o en el borde del tile; no distingue quién es grande y quién chico cuando hay desigualdad.
- **Uso**: fallback cuando el blob no es confiable, o como cross-check de A.

### C. Combinación de A + B

- **Qué**: promedio ponderado (más peso al heatmap), recortado a la banda de la finca.
- **A favor**: reduce la varianza de cada estimador por separado.
- **En contra**: dos parámetros más que calibrar.
- **Uso**: si A sola no baja los rechazos lo suficiente.

### D. Box elíptico desde los momentos del blob

- **Qué**: calcular la covarianza de los píxeles del blob → ejes de la elipse → box con ancho/alto distintos.
- **A favor**: ataca directamente el hallazgo `r_enclose/r_eq` mediana 1.32: las copas no son círculos.
- **En contra**: más sensible al ruido y a coronas vecinas; requiere validar la geometría.
- **Uso**: siguiente paso después de A si siguen apareciendo copas parcialmente cubiertas.

### E. Dos pasadas con SAM

- **Qué**: primera pasada con el radio mediano → máscara → `r = √(área_máscara/π)` → segunda pasada con el box corregido.
- **A favor**: usa la máscara real, que es lo que finalmente se publica.
- **En contra**: duplica el costo de SAM; si la primera pasada sobre-segmenta, el feedback empeora. Riesgo de realimentación.
- **Uso**: experimento acotado, no primera opción.

### F. Reentrenar la cabeza de radio

- **Qué**: corregir `radius_loss` (comparar contra el target y hacer `squeeze(1)` para evitar el broadcasting), supervisar el radio en los píxeles del núcleo y reentrenar en Kaggle T4. **Corrección 2026-11-19**: la causa del colapso fue este bug, no la augmentación.
- **A favor**: es la solución "correcta": el modelo aprendería el tamaño junto con el centro.
- **En contra**: otra corrida de ~12 min y nueva validación. El riesgo de colapso ya se puede descartar antes con el test local de sobreajuste.
- **Uso**: si A–D no alcanzan el objetivo.

## 4. Recomendación

**Implementar A primero**, con B como fallback y recorte por finca. Es gratis, usa los pesos congelados y ataca la causa directa del 77% de los rechazos. Después medir; si quedan copas mal cubiertas, sumar D (elipse). Recién entonces considerar F.

No cambiar el modelo v1: generar una **v1.1 del decode/cascada** con el radio estimado, dejando `data/centroides/v1/model.pt` intacto.

## 5. Cómo validar el cambio

| Métrica | Línea base | Criterio |
|---|---|---|
| Rechazos por `area` en 9 auditados | 65 | reducir al menos 30% |
| Cobertura del box (polígono dentro de `±1.15·r_est`) | 10.2% | subir de forma medible |
| F1 de instancia dev / holdout | 0.869 / 0.799 | no bajar más de 0.002 |
| Recall de propuesta en frames CVAT no vistos | Yulima 0.86–0.93, Villas 0.33–0.80 | no empeorar |

Todo se mide con la métrica unificada (instance F1, IoU ≥ 0.5, one-to-one) en dev y holdout limpios. El holdout 20 no se toca para ajustar.

## 6. Resultado de la opción A (v1.1) — 2026-11-18

Implementada en el kernel `marcosalvarez234/limon-centroides-cascade-v11`, sin tocar los pesos v1. Radio por copa desde el blob del heatmap, NN calculado sobre los sobrevivientes del NMS como fallback y recorte a `[0.6, 1.6]·mediana de la finca`.

| Métrica | Línea base | v1.1 | Delta |
|---|---:|---:|---:|
| F1 dev | 0.8694 | **0.9241** | +0.0547 |
| F1 holdout | 0.7990 | **0.8938** | +0.0948 |
| Precisión holdout | 0.8743 | **0.9188** | +0.0445 |
| Rechazos por `area` | 65 | **27** | −58.5% |
| Rechazos por `forma` | 7 | 2 | −5 |
| Rechazos por `conteo` | 10 | 15 | +5 |
| Rechazos por `nms` | 2 | 0 | −2 |

Los tres gates E2/E3 siguen pasando y con más margen. El aumento de rechazos por `conteo` es esperable: al aceptar más copas, la alarma de conteo poda más excesos; no afectó la precisión.

Bug corregido durante la implementación: el fallback de vecinos se calculaba sobre los picos **antes** del NMS; los duplicados acortaban todas las distancias y los radios caían al piso. Se reordenó a NMS → vecinos → recorte.

Detalle en `data/centroides/cascade_v11/comparacion.json`.

## 8. Barrido de descompresión (2026-11-19)

Sobre las mismas 100 primeras frames de Villas (813 pares emparejados), se barrió el recorte y la regla de combinación blob/vecino. Radio real en filas, estimado en columnas, bins `<40 / 40-65 / ≥65`.

| Config | Recorte | Regla | Acierto pequeñas | medianas | grandes | IoU med | IoU≥0.5 | Err med |
|---|---|---|---:|---:|---:|---:|---:|---:|
| C0 base | 0.6–1.6 | blob + guard | 67.7% | 74.8% | 2.7% | 0.658 | 0.748 | 0.206 |
| C1 ancho | 0.4–2.2 | blob + guard | 67.7% | 74.8% | 2.7% | 0.662 | 0.749 | 0.204 |
| C2 ancho sin guard | 0.4–2.2 | blob | 59.5% | **94.5%** | 5.1% | 0.702 | 0.982 | 0.170 |
| C3 max | 0.4–2.2 | máx(blob, vecino) | 10.1% | 44.9% | 35.0% | 0.606 | 0.659 | 0.261 |
| C4 blend | 0.4–2.2 | 0.7·blob + 0.3·vecino | 24.7% | 59.3% | 19.7% | 0.640 | 0.734 | 0.238 |
| C5 boost | 0.4–2.2 | blob×1.3 si blob≥50 | 59.5% | 15.0% | **100%** | **0.754** | 0.973 | 0.173 |
| C6 vecino | 0.4–2.2 | vecino/2 | 20.3% | 21.6% | 32.7% | 0.369 | 0.355 | 0.535 |

**Conclusión: no adoptar ninguna config del barrido como v1.2.** Ninguna mejora los tres bins a la vez: C5 arregla las copas grandes (100%) pero destruye las medianas (15%); C3 mejora grandes pero empeora el resto; C2 es básicamente la línea base con el guard quitado. El error mediano se mueve apenas (0.206 → 0.170).

La compresión no es un artefacto del recorte ni del guard: es **estructural**. El `σ` del target está saturado en 16 px, así que el modelo nunca vio blobs de copas grandes y no tiene señal para distinguir "mediana" de "muy grande". Ningún postproceso puede inventar esa información. Para mejorar los extremos hace falta otra señal (extensión de vegetación local) o reentrenar sin la saturación.

**Caveat metodológico**: el harness del barrido es una réplica del `decode` de la v1.1 y no lo reproduce byte a byte — en un control puntual, 7 de 8 radios coincidieron y uno difirió. Los números sirven para comparar configuraciones entre sí, no como métrica absoluta. Cualquier adopción real exige correr E2/E3 en el kernel de Kaggle.

Archivos: `data/centroides/test100/barrido_radio.json`, `matriz_radio.png`, `confusion.json`.

## 9. Log

- 2026-11-18: análisis geométrico de 4602 copas y calibración del blob en 40 frames. Factor empírico 2.17·4, error mediano 15.8%, Spearman 0.70–0.77. Documentado sin reentrenar.
- 2026-11-18: usuario dijo "procede con A". Implementado y medido: adopción de la v1.1 de la cascada.
- 2026-11-19: barrido de descompresión en 100 frames. Sin ganador claro; se documenta la causa estructural y se descarta la v1.2 por postproceso.
