# Matriz de confusión — StarDist vs gold_v2

Modelo: `marcosalvarez234/limon-stardist-v1-weights` · dataset: `/kaggle/input/datasets/marcosalvarez234/limon-gold-v2d` · `prob_thresh=0.5929` · matching de instancias IoU ≥ 0.5.

Imágenes: 405 (LasVillas, Magdala). Splits: {'train': 346, 'dev': 38, 'holdout': 20, 'audited_excluded': 9}.

**Matriz de píxeles**: filas = clase real, columnas = clase predicha. Se construye con la unión de máscaras de copa por imagen (fondo = el resto del tile 512×512). **Instancias**: TP/FP/FN por matching greedy IoU ≥ 0.5.

> Ojo: `train` son imágenes vistas durante el entrenamiento; `dev`/`holdout` son las honestas. Las filas `farm:*` mezclan todos los splits.

## nms_0.02

| Grupo | TP | FP | FN | Precision | Recall | F1 | IoU píxel |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dev (no visto) | 259 | 48 | 53 | 0.8436 | 0.8301 | 0.8368 | 0.7281 |
| Holdout (no visto) | 138 | 29 | 32 | 0.8263 | 0.8118 | 0.8190 | 0.7110 |
| LasVillas (todas) | 1703 | 425 | 267 | 0.8003 | 0.8645 | 0.8311 | 0.7234 |
| Magdala (todas) | 1364 | 238 | 440 | 0.8514 | 0.7561 | 0.8009 | 0.6898 |
| LasVillas / dev | 166 | 31 | 24 | 0.8426 | 0.8737 | 0.8579 | 0.7699 |
| LasVillas / holdout | 77 | 15 | 9 | 0.8370 | 0.8953 | 0.8652 | 0.7525 |
| Magdala / dev | 93 | 17 | 29 | 0.8455 | 0.7623 | 0.8017 | 0.6320 |
| Magdala / holdout | 61 | 14 | 23 | 0.8133 | 0.7262 | 0.7673 | 0.6402 |
| Train (visto) | 2668 | 585 | 621 | 0.8202 | 0.8112 | 0.8157 | 0.7093 |
| Todas | 3067 | 663 | 707 | 0.8223 | 0.8127 | 0.8174 | 0.7109 |

| Grupo | Matriz de píxeles (TP/FP · FN/TN) |
|---|---|
| Dev (no visto) | `2,399,192` / `434,516` · `461,346` / `6,666,418` |
| Holdout (no visto) | `1,251,415` / `267,193` · `241,370` / `3,482,902` |
| LasVillas (todas) | `18,172,487` / `3,970,891` · `2,976,653` / `35,959,521` |
| Magdala (todas) | `10,258,500` / `1,992,872` · `2,620,993` / `30,216,403` |
| LasVillas / dev | `1,767,791` / `263,206` · `265,030` / `3,471,141` |
| LasVillas / holdout | `835,393` / `144,947` · `129,814` / `1,511,286` |
| Magdala / dev | `631,401` / `171,310` · `196,316` / `3,195,277` |
| Magdala / holdout | `416,022` / `122,246` · `111,556` / `1,971,616` |

## nms_0.1

| Grupo | TP | FP | FN | Precision | Recall | F1 | IoU píxel |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dev (no visto) | 272 | 53 | 40 | 0.8369 | 0.8718 | 0.8540 | 0.7488 |
| Holdout (no visto) | 144 | 31 | 26 | 0.8229 | 0.8471 | 0.8348 | 0.7371 |
| LasVillas (todas) | 1757 | 452 | 213 | 0.7954 | 0.8919 | 0.8409 | 0.7344 |
| Magdala (todas) | 1481 | 262 | 323 | 0.8497 | 0.8210 | 0.8351 | 0.7381 |
| LasVillas / dev | 170 | 31 | 20 | 0.8458 | 0.8947 | 0.8696 | 0.7828 |
| LasVillas / holdout | 77 | 16 | 9 | 0.8280 | 0.8953 | 0.8603 | 0.7502 |
| Magdala / dev | 102 | 22 | 20 | 0.8226 | 0.8361 | 0.8293 | 0.6723 |
| Magdala / holdout | 67 | 15 | 17 | 0.8171 | 0.7976 | 0.8072 | 0.7151 |
| Train (visto) | 2820 | 629 | 469 | 0.8176 | 0.8574 | 0.8370 | 0.7346 |
| Todas | 3238 | 714 | 536 | 0.8193 | 0.8580 | 0.8382 | 0.7358 |

| Grupo | Matriz de píxeles (TP/FP · FN/TN) |
|---|---|
| Dev (no visto) | `2,483,709` / `456,189` · `376,829` / `6,644,745` |
| Holdout (no visto) | `1,305,667` / `278,461` · `187,118` / `3,471,634` |
| LasVillas (todas) | `18,651,422` / `4,246,463` · `2,497,718` / `35,683,949` |
| Magdala (todas) | `11,080,873` / `2,132,977` · `1,798,620` / `30,076,298` |
| LasVillas / dev | `1,798,385` / `264,527` · `234,436` / `3,469,820` |
| LasVillas / holdout | `835,204` / `148,137` · `130,003` / `1,508,096` |
| Magdala / dev | `685,324` / `191,662` · `142,393` / `3,174,925` |
| Magdala / holdout | `470,463` / `130,324` · `57,115` / `1,963,538` |

