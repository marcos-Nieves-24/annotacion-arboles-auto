# Magdala — copas que se escapan a StarDist

Modelo `limon-stardist-v1` · `prob_thresh=0.5929` · matching IoU ≥ 0.5 · 26 imágenes Magdala de dev/holdout.

**Sin escritura en CVAT.** Los FN son copas del oro que StarDist no alcanzó a detectar.

## NMS 0.02

TP 154 · FP 31 · **FN 52**

| Diagnóstico | Valor |
|---|---:|
| No detectadas (`best IoU < 0.1`) | 52 (100.0%) |
| Sub-umbral (`0.1 ≤ best IoU < 0.5`) | 0 (0.0%) |
| Tocan borde del tile (≤8 px) | 37 (71.2%) |
| Área mediana (px) | 2497 |
| Área p25 / p75 | 1157 / 7775 |
| Mejor IoU mediano | 0.001 |
| Distancia mediana a la predicción más cercana (px) | 94.1 |

| Borde del tile | Copas perdidas |
|---|---:|
| arriba | 9 (17.3%) |
| abajo | 16 (30.8%) |
| izquierda | 3 (5.8%) |
| derecha | 10 (19.2%) |
| centro del tile | 15 (28.8%) |

| Imagen | split | FN |
|---|---|---:|
| `Limon_Magdala_01027.png` | dev | 6 |
| `Limon_Magdala_01010.png` | holdout | 5 |
| `Limon_Magdala_01043.png` | holdout | 5 |
| `Limon_Magdala_00866.png` | holdout | 4 |
| `Limon_Magdala_00946.png` | holdout | 4 |
| `Limon_Magdala_01035.png` | dev | 4 |
| `Limon_Magdala_01047.png` | dev | 4 |
| `Limon_Magdala_01199.png` | dev | 4 |

| Imagen | split | ann_id | centroide (x,y) | área px | mejor IoU | tipo | borde |
|---|---|---:|---|---:|---:|---|---|
| `Limon_Magdala_01043.png` | holdout | 3369 | (441.9, 291.0) | 11397 | 0.090 | not_detected | sí |
| `Limon_Magdala_01012.png` | dev | 3001 | (370.2, 37.0) | 7612 | 0.070 | not_detected | sí |
| `Limon_Magdala_01042.png` | dev | 3340 | (262.6, 38.8) | 8191 | 0.069 | not_detected | sí |
| `Limon_Magdala_01010.png` | holdout | 2988 | (357.0, 498.8) | 2792 | 0.064 | not_detected | sí |
| `Limon_Magdala_01047.png` | dev | 3437 | (74.7, 274.6) | 8243 | 0.061 | not_detected | no |
| `Limon_Magdala_01012.png` | dev | 3004 | (467.6, 373.4) | 11987 | 0.060 | not_detected | sí |
| `Limon_Magdala_01027.png` | dev | 3145 | (487.4, 433.6) | 5436 | 0.056 | not_detected | sí |
| `Limon_Magdala_01027.png` | dev | 3148 | (164.6, 261.4) | 10083 | 0.049 | not_detected | no |
| `Limon_Magdala_01010.png` | holdout | 2971 | (192.5, 52.5) | 10424 | 0.034 | not_detected | sí |
| `Limon_Magdala_01035.png` | dev | 3267 | (490.6, 173.1) | 4840 | 0.034 | not_detected | sí |
| `Limon_Magdala_01010.png` | holdout | 2968 | (58.8, 200.1) | 13350 | 0.028 | not_detected | sí |
| `Limon_Magdala_01043.png` | holdout | 3358 | (4.6, 272.0) | 738 | 0.028 | not_detected | sí |
| `Limon_Magdala_00866.png` | holdout | 2288 | (275.2, 405.0) | 7656 | 0.024 | not_detected | no |
| `Limon_Magdala_01035.png` | dev | 3269 | (393.7, 100.6) | 12262 | 0.022 | not_detected | no |
| `Limon_Magdala_01199.png` | dev | 4035 | (164.7, 33.9) | 6656 | 0.021 | not_detected | sí |
| `Limon_Magdala_01199.png` | dev | 4025 | (500.3, 58.4) | 1676 | 0.019 | not_detected | sí |
| `Limon_Magdala_01042.png` | dev | 3343 | (239.7, 485.9) | 4349 | 0.019 | not_detected | sí |
| `Limon_Magdala_01042.png` | dev | 3337 | (455.3, 8.2) | 1433 | 0.019 | not_detected | sí |
| `Limon_Magdala_01010.png` | holdout | 2985 | (314.1, 133.8) | 11397 | 0.014 | not_detected | no |
| `Limon_Magdala_01027.png` | dev | 3141 | (199.4, 450.3) | 13149 | 0.012 | not_detected | sí |
| `Limon_Magdala_01255.png` | dev | 4349 | (11.7, 174.7) | 1534 | 0.011 | not_detected | sí |
| `Limon_Magdala_01255.png` | dev | 4359 | (344.6, 195.3) | 8207 | 0.009 | not_detected | no |
| `Limon_Magdala_01010.png` | holdout | 2979 | (317.9, 386.2) | 6628 | 0.008 | not_detected | no |
| `Limon_Magdala_01043.png` | holdout | 3366 | (265.9, 35.8) | 5051 | 0.005 | not_detected | sí |
| `Limon_Magdala_01199.png` | dev | 4030 | (385.8, 465.3) | 4604 | 0.003 | not_detected | sí |
| `Limon_Magdala_01047.png` | dev | 3442 | (418.5, 329.0) | 2280 | 0.001 | not_detected | no |
| `Limon_Magdala_01027.png` | dev | 3147 | (365.5, 349.9) | 10108 | 0.001 | not_detected | no |
| `Limon_Magdala_00544.png` | holdout | 2148 | (264.1, 503.2) | 847 | 0.000 | not_detected | sí |
| `Limon_Magdala_00628.png` | dev | 2215 | (249.6, 368.9) | 2714 | 0.000 | not_detected | no |
| `Limon_Magdala_00628.png` | dev | 2213 | (183.1, 15.6) | 782 | 0.000 | not_detected | sí |
| `Limon_Magdala_00806.png` | holdout | 2246 | (66.5, 498.9) | 1467 | 0.000 | not_detected | sí |
| `Limon_Magdala_00806.png` | holdout | 2247 | (121.6, 393.3) | 1267 | 0.000 | not_detected | no |
| `Limon_Magdala_00866.png` | holdout | 2300 | (195.1, 202.5) | 4723 | 0.000 | not_detected | no |
| `Limon_Magdala_00866.png` | holdout | 2287 | (231.5, 500.8) | 1688 | 0.000 | not_detected | sí |
| `Limon_Magdala_00866.png` | holdout | 2297 | (380.0, 6.6) | 760 | 0.000 | not_detected | sí |
| `Limon_Magdala_00889.png` | holdout | 2411 | (506.1, 237.2) | 630 | 0.000 | not_detected | sí |
| `Limon_Magdala_00946.png` | holdout | 2587 | (402.7, 504.3) | 959 | 0.000 | not_detected | sí |
| `Limon_Magdala_00946.png` | holdout | 2586 | (71.6, 504.3) | 748 | 0.000 | not_detected | sí |
| `Limon_Magdala_00946.png` | holdout | 2589 | (504.4, 481.6) | 598 | 0.000 | not_detected | sí |
| `Limon_Magdala_00946.png` | holdout | 2588 | (504.6, 432.7) | 516 | 0.000 | not_detected | sí |
| `Limon_Magdala_00961.png` | holdout | 2698 | (194.0, 505.2) | 467 | 0.000 | not_detected | sí |
| `Limon_Magdala_01012.png` | dev | 2998 | (175.9, 490.7) | 2271 | 0.000 | not_detected | sí |
| `Limon_Magdala_01027.png` | dev | 3152 | (483.4, 81.5) | 8133 | 0.000 | not_detected | sí |
| `Limon_Magdala_01027.png` | dev | 3138 | (324.0, 487.4) | 6606 | 0.000 | not_detected | sí |
| `Limon_Magdala_01035.png` | dev | 3266 | (216.2, 71.5) | 1560 | 0.000 | not_detected | no |
| `Limon_Magdala_01035.png` | dev | 3264 | (66.3, 488.4) | 798 | 0.000 | not_detected | sí |
| `Limon_Magdala_01043.png` | holdout | 3355 | (498.9, 264.3) | 1892 | 0.000 | not_detected | sí |
| `Limon_Magdala_01043.png` | holdout | 3356 | (126.3, 502.3) | 1224 | 0.000 | not_detected | sí |
| `Limon_Magdala_01047.png` | dev | 3456 | (409.7, 11.7) | 1282 | 0.000 | not_detected | sí |
| `Limon_Magdala_01047.png` | dev | 3447 | (441.5, 475.9) | 641 | 0.000 | not_detected | no |
| `Limon_Magdala_01199.png` | dev | 4021 | (332.6, 505.3) | 654 | 0.000 | not_detected | sí |
| `Limon_Magdala_01255.png` | dev | 4358 | (263.1, 292.4) | 1975 | 0.000 | not_detected | no |

## NMS 0.1

TP 169 · FP 37 · **FN 37**

| Diagnóstico | Valor |
|---|---:|
| No detectadas (`best IoU < 0.1`) | 35 (94.6%) |
| Sub-umbral (`0.1 ≤ best IoU < 0.5`) | 2 (5.4%) |
| Tocan borde del tile (≤8 px) | 28 (75.7%) |
| Área mediana (px) | 1560 |
| Área p25 / p75 | 782 / 4723 |
| Mejor IoU mediano | 0.000 |
| Distancia mediana a la predicción más cercana (px) | 183.5 |

| Borde del tile | Copas perdidas |
|---|---:|
| arriba | 7 (18.9%) |
| abajo | 14 (37.8%) |
| izquierda | 1 (2.7%) |
| derecha | 7 (18.9%) |
| centro del tile | 9 (24.3%) |

| Imagen | split | FN |
|---|---|---:|
| `Limon_Magdala_00946.png` | holdout | 4 |
| `Limon_Magdala_01027.png` | dev | 4 |
| `Limon_Magdala_00866.png` | holdout | 3 |
| `Limon_Magdala_01012.png` | dev | 3 |
| `Limon_Magdala_01042.png` | dev | 3 |
| `Limon_Magdala_01043.png` | holdout | 3 |
| `Limon_Magdala_01047.png` | dev | 3 |
| `Limon_Magdala_00628.png` | dev | 2 |

| Imagen | split | ann_id | centroide (x,y) | área px | mejor IoU | tipo | borde |
|---|---|---:|---|---:|---:|---|---|
| `Limon_Magdala_01027.png` | dev | 3153 | (478.2, 250.4) | 8886 | 0.177 | sub_threshold | sí |
| `Limon_Magdala_01043.png` | holdout | 3355 | (498.9, 264.3) | 1892 | 0.119 | sub_threshold | sí |
| `Limon_Magdala_01012.png` | dev | 3001 | (370.2, 37.0) | 7612 | 0.070 | not_detected | sí |
| `Limon_Magdala_01042.png` | dev | 3340 | (262.6, 38.8) | 8191 | 0.069 | not_detected | sí |
| `Limon_Magdala_01010.png` | holdout | 2988 | (357.0, 498.8) | 2792 | 0.064 | not_detected | sí |
| `Limon_Magdala_01012.png` | dev | 3004 | (467.6, 373.4) | 11987 | 0.060 | not_detected | sí |
| `Limon_Magdala_01043.png` | holdout | 3358 | (4.6, 272.0) | 738 | 0.028 | not_detected | sí |
| `Limon_Magdala_01199.png` | dev | 4035 | (164.7, 33.9) | 6656 | 0.021 | not_detected | sí |
| `Limon_Magdala_01042.png` | dev | 3343 | (239.7, 485.9) | 4349 | 0.019 | not_detected | sí |
| `Limon_Magdala_01042.png` | dev | 3337 | (455.3, 8.2) | 1433 | 0.019 | not_detected | sí |
| `Limon_Magdala_01010.png` | holdout | 2979 | (317.9, 386.2) | 6628 | 0.008 | not_detected | no |
| `Limon_Magdala_01047.png` | dev | 3442 | (418.5, 329.0) | 2280 | 0.001 | not_detected | no |
| `Limon_Magdala_01027.png` | dev | 3147 | (365.5, 349.9) | 10108 | 0.001 | not_detected | no |
| `Limon_Magdala_00544.png` | holdout | 2148 | (264.1, 503.2) | 847 | 0.000 | not_detected | sí |
| `Limon_Magdala_00628.png` | dev | 2215 | (249.6, 368.9) | 2714 | 0.000 | not_detected | no |
| `Limon_Magdala_00628.png` | dev | 2213 | (183.1, 15.6) | 782 | 0.000 | not_detected | sí |
| `Limon_Magdala_00806.png` | holdout | 2246 | (66.5, 498.9) | 1467 | 0.000 | not_detected | sí |
| `Limon_Magdala_00806.png` | holdout | 2247 | (121.6, 393.3) | 1267 | 0.000 | not_detected | no |
| `Limon_Magdala_00866.png` | holdout | 2300 | (195.1, 202.5) | 4723 | 0.000 | not_detected | no |
| `Limon_Magdala_00866.png` | holdout | 2287 | (231.5, 500.8) | 1688 | 0.000 | not_detected | sí |
| `Limon_Magdala_00866.png` | holdout | 2297 | (380.0, 6.6) | 760 | 0.000 | not_detected | sí |
| `Limon_Magdala_00889.png` | holdout | 2411 | (506.1, 237.2) | 630 | 0.000 | not_detected | sí |
| `Limon_Magdala_00946.png` | holdout | 2587 | (402.7, 504.3) | 959 | 0.000 | not_detected | sí |
| `Limon_Magdala_00946.png` | holdout | 2586 | (71.6, 504.3) | 748 | 0.000 | not_detected | sí |
| `Limon_Magdala_00946.png` | holdout | 2589 | (504.4, 481.6) | 598 | 0.000 | not_detected | sí |
| `Limon_Magdala_00946.png` | holdout | 2588 | (504.6, 432.7) | 516 | 0.000 | not_detected | sí |
| `Limon_Magdala_00961.png` | holdout | 2698 | (194.0, 505.2) | 467 | 0.000 | not_detected | sí |
| `Limon_Magdala_01012.png` | dev | 2998 | (175.9, 490.7) | 2271 | 0.000 | not_detected | sí |
| `Limon_Magdala_01027.png` | dev | 3152 | (483.4, 81.5) | 8133 | 0.000 | not_detected | sí |
| `Limon_Magdala_01027.png` | dev | 3138 | (324.0, 487.4) | 6606 | 0.000 | not_detected | sí |
| `Limon_Magdala_01035.png` | dev | 3266 | (216.2, 71.5) | 1560 | 0.000 | not_detected | no |
| `Limon_Magdala_01035.png` | dev | 3264 | (66.3, 488.4) | 798 | 0.000 | not_detected | sí |
| `Limon_Magdala_01043.png` | holdout | 3356 | (126.3, 502.3) | 1224 | 0.000 | not_detected | sí |
| `Limon_Magdala_01047.png` | dev | 3456 | (409.7, 11.7) | 1282 | 0.000 | not_detected | sí |
| `Limon_Magdala_01047.png` | dev | 3447 | (441.5, 475.9) | 641 | 0.000 | not_detected | no |
| `Limon_Magdala_01199.png` | dev | 4021 | (332.6, 505.3) | 654 | 0.000 | not_detected | sí |
| `Limon_Magdala_01255.png` | dev | 4358 | (263.1, 292.4) | 1975 | 0.000 | not_detected | no |

## Lectura

- `no detectada`: no hay ninguna predicción con IoU ≥ 0.1 → el modelo no vio la copa.
- `sub_threshold`: existe una predicción solapada pero no llega a IoU 0.5 → forma/recorte incorrecto.
- **Ubicación**: la mayoría de las pérdidas está en el borde del tile (arriba/abajo/derecha), no en el centro. El mapa de calor `fn_heatmap_nms*.png` lo muestra: franjas densas en y≈0 e y≈500.
- El `best IoU` mediano es ~0 y la distancia a la predicción más cercana es ~180 px: no son copas mal recortadas ni fusionadas con vecinas, **son copas que StarDist no detecta**.
- El área mediana de las perdidas (~1500 px) descarta que sean solo copas chicas.
- Los overlays `fn_nms*_*.png` marcan en rojo cada FN con su `ann_id`; verde = oro, amarillo = TP, azul = predicción no match.

## Corrección — "truncadas" no era correcto

La hipótesis inicial decía que las copas perdidas estaban **truncadas** y que el entrenamiento las había excluido como `iscrowd`. **Eso es falso**, verificado en `experimentos/yolo/scripts/convert_gold.py`:

```python
BORDER_TOL = 2.0
BORDER_MIN_VERTICES = 3   # iscrowd sólo si >=3 vértices en el borde
# exactamente 2 usualmente es un recorte leve (entra/sale), entrenable
```

Aplicando esa misma regla a las 37 FN de NMS 0.1, usando los polígonos del `coco.json`:

| Vértices en el borde | FN | `iscrowd` |
|---:|---:|---:|
| 0 | 10 | 0 |
| 2 | 27 | 0 |
| ≥3 (sí serían "truncadas") | **0** | 0 |

Ninguna FN cumple la definición del conversor de copa truncada. Las 27 que tocan el borde tienen **exactamente 2 vértices**: entran y salen del tile, y el conversor las conservó como entrenables.

Tasa de fallo real en Magdala dev/holdout (184 copas del oro no-iscrowd):

| Vértices en borde | Copas | FN | Tasa de fallo |
|---:|---:|---:|---:|
| 0 (interiores) | 92 | 10 | 10.9% |
| 2 (cruzan el borde) | 91 | 27 | **29.7%** |

Conclusión corregida: no es un problema de copas truncadas excluidas del entrenamiento. Es que **las copas que cruzan el borde del tile fallan ~2.7× más** que las interiores — la causa sigue siendo el borde, pero el mecanismo es distinto del que se hipotetizó.

Nota: el campo `bbox` del `coco.json` está en 0 para estas anotaciones; las coordenadas de este análisis vienen de rasterizar el `segmentation`, no de ese campo.

