# Reentrenamiento del radio y flag de vegetación — 2026-11-19

Alcance: evaluar las opciones 2 y 3 propuestas por el usuario tras la matriz de radio. **Conclusión: se mantiene la v1.1 (blob) como cascada canónica.** Los pesos v1 no se modificaron.

## 1. Corrección de diagnóstico

Se había atribuido el colapso de la cabeza de radio a la augmentación afín. **Era incorrecto.** La causa real estaba en `radius_loss` del kernel v1:

```python
diff = F.smooth_l1_loss(pred, torch.zeros_like(pred), reduction="none")  # contra CEROS
return (diff * w).sum() / max(w.sum().item(), 1.0)                       # (B,1,128,128) vs (B,128,128)
```

Dos bugs: (1) comparaba contra ceros, enseñando a predecir 0; (2) broadcasting `(B,1,H,W)` vs `(B,H,W)` → `(B,B,H,W)`. Corregido en todos los kernels nuevos con `pred.squeeze(1)` y `target`.

## 2. Test local previo (antes de quemar Kaggle)

8 frames, 40 pasos de sobreajuste con la pérdida corregida: pérdida de radio **64.1 → 2.2**, radio predicho mediano **63.3 px** contra **65 px** real. La cabeza sí aprende.

## 3. Corridas en Kaggle T4

| Kernel | Cambio | Radio (error med / rango) | Recall propuesta dev / auditados |
|---|---|---|---|
| `limon-centroides-v2-radio` | pérdida corregida, entrenamiento conjunto | **10.1%**, 15–**91 px** | 0.645 / 0.741 ❌ |
| `limon-centroides-v3-head` | v1 congelado, solo cabeza de radio, 30 ép | 27.2% (dev), 26–**93 px** | 0.678 / 0.800 (con NMS del decode) |

Hallazgo clave: el recall bajo de v2/v3 **no** era del heatmap. Al usar el radio predicho en el NMS (`0.7·(r_i+r_j)` con radios ~60–90 px) se suprimían copas vecinas. Con NMS fijo de 40 px sobre v3, el recall vuelve a **0.824** (v1: 0.826). El heatmap de v3 es el de v1, intacto.

## 4. Calibración del radio v3 en 100 frames

Ajuste afín sobre 806 pares: `r_real ≈ 0.977·r_pred − 9.52`.

| Calibración | Pequeñas | Medianas | Grandes | Error med |
|---|---:|---:|---:|---:|
| cruda | 3.2% | 43.1% | **83.3%** | 0.208 |
| afín (0.977·x−9.52) | 40.4% | 79.3% | 43.7% | 0.138 |
| blob v1.1 (referencia) | 59.5% | **94.5%** | 5.1% | 0.173 |

La cabeza entrenada **sí rompe el techo de 64 px** y mejora el error global, pero reparte peor los bins: gana grandes, pierde medianas y pequeñas.

## 5. Flag de vegetación calibrado

Extensión verde local alrededor del centroide, con filtros de compacidad (`fill > 0.5`) y centrado (`dist < 15 px`). Sobre los 813 pares:

| Regla | Pequeñas | Medianas | Grandes | Error med |
|---|---:|---:|---:|---:|
| A base (blob) | 59.5% | 94.5% | 5.1% | 0.173 |
| flag calibrado → r=75 | 59.5% | 94.2% | 10.5% | 0.168 |

Mejora real pero pequeña; no justifica adopción sola.

## 6. Test decisivo: cascada v2.1 con radio entrenado

Kernel `limon-centroides-cascade-v21`: pesos v3, NMS fijo 40, radio afín, clip `[0.4, 2.2]×mediana`. Resultado en Kaggle:

| Versión | F1 dev | F1 holdout | P holdout | Rechazos `area` |
|---|---:|---:|---:|---:|
| base (blob fijo) | 0.8694 | 0.7990 | 0.8743 | 65 |
| **v1.1 (blob calibrado)** | **0.9241** | **0.8938** | **0.9188** | **27** |
| v2.1 (cabeza entrenada) | 0.9058 | 0.8722 | 0.9110 | 31 |

Los tres gates pasan en v2.1, pero **es peor que v1.1 en todo**. Se descarta.

## 7. Decisión

- **Mantener v1.1** (blob calibrado + vecino + recorte) como cascada canónica.
- El reentrenamiento sirvió para: (a) corregir un bug real y documentarlo, (b) demostrar que la cabeza puede predecir radios de hasta ~93 px, (c) identificar que el NMS con radio predicho era el culpable del recall bajo.
- No se reentrena más: el costo adicional ya no devuelve F1.
- Si en el futuro aparece un dominio con copas mucho más grandes, la cabeza v3 es una alternativa a evaluar, no la v1.1.

## 8. Selección informada de máscaras y Voronoi (2026-11-19)

Sobre la base v1.1 se probaron dos cambios: (a) elegir entre las 3 máscaras de SAM por cercanía del centroide a `(cx, cy)` y del área a `π·r²` en vez del score de SAM; (b) asignar los píxeles en disputa entre máscaras al centroide más cercano (Voronoi local).

| Kernel | Cambio | F1 dev | F1 holdout | Rechazos `area` |
|---|---|---:|---:|---:|
| `cascade-v11` | ninguno (adoptada) | **0.9241** | **0.8938** | 27 |
| `cascade-v30` | ambos | 0.8543 | 0.8265 | 55 |
| `cascade-v31` | solo selección informada | 0.9155 | 0.8772 | — |
| `cascade-v32` | solo Voronoi | 0.8300 | 0.8312 | — |

**Conclusión: ninguno mejora. Se mantiene v1.1.**

- Voronoi es claramente dañino: ~1300 píxeles disputados por frame; al recortar esas zonas se achican las máscaras y se duplican los rechazos por área (55 vs 27). Las copas que se tocan comparten mucho territorio, y la asignación por centro más cercano destruye lóbulos reales.
- La selección informada es casi neutral (-0.017 F1 en holdout): el score propio de SAM ya elegía razonablemente bien. No justifica el cambio.

Los píxeles en disputa son altos porque las máscaras de copas vecinas se solapan mucho en el oro; eso es consistente con el hallazgo de que 68% de los frames tienen copas que se tocan. Para ese problema haría falta una separación por flujo (FG-TreeSeg) o watershed con marcadores, no una poda geométrica dura.

Artefactos: `data/centroides/cascade_v30/`, `cascade_v31/`, `cascade_v32/` con reportes, reviews y overlays.

## 9. Artefactos

- `data/centroides/v2_radio/`: pesos y reporte de la corrida conjunta.
- `data/centroides/v3_head/`: pesos congelados de v1 + cabeza entrenada, `calibracion.json`.
- `data/centroides/flag_vegetacion/pares_veg.json`.
- `data/centroides/cascade_v21/`: reporte y revisión de la cascada v2.1.
- Kernels: `limon-centroides-v2-radio`, `limon-centroides-v3-head`, `limon-centroides-cascade-v21`.
