# Barrido de `prob_thresh` — StarDist en Magdala dev/holdout

Modelo `limon-stardist-v1` · 26 imágenes · matching IoU ≥ 0.5 · baseline `prob_thresh=0.5929`.

Mismo set usado en el análisis de FN (26 imágenes), para comparar directamente contra las 37 FN del baseline.

## NMS 0.02

| prob_thresh | TP | FP | FN | Precision | Recall | F1 | FN borde 0/1/2/≥3 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.3000 | 182 | 112 | 24 | 0.6190 | 0.8835 | 0.7280 | 11/0/13/0 |
| 0.3500 | 181 | 102 | 25 | 0.6396 | 0.8786 | 0.7403 | 12/0/13/0 |
| 0.4000 | 177 | 91 | 29 | 0.6604 | 0.8592 | 0.7468 | 13/0/16/0 |
| 0.4500 | 172 | 70 | 34 | 0.7107 | 0.8350 | 0.7679 | 15/0/19/0 |
| 0.5000 | 166 | 57 | 40 | 0.7444 | 0.8058 | 0.7739 | 15/0/25/0 |
| 0.5500 | 159 | 44 | 47 | 0.7833 | 0.7718 | 0.7775 | 16/0/31/0 |
| 0.5929 | 154 | 31 | 52 | 0.8324 | 0.7476 | 0.7877 | 17/0/35/0 |

## NMS 0.1

| prob_thresh | TP | FP | FN | Precision | Recall | F1 | FN borde 0/1/2/≥3 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.3000 | 193 | 121 | 13 | 0.6146 | 0.9369 | 0.7423 | 5/0/8/0 |
| 0.3500 | 192 | 110 | 14 | 0.6358 | 0.9320 | 0.7559 | 6/0/8/0 |
| 0.4000 | 190 | 99 | 16 | 0.6574 | 0.9223 | 0.7677 | 7/0/9/0 |
| 0.4500 | 185 | 80 | 21 | 0.6981 | 0.8981 | 0.7856 | 8/0/13/0 |
| 0.5000 | 180 | 64 | 26 | 0.7377 | 0.8738 | 0.8000 | 8/0/18/0 |
| 0.5500 | 174 | 51 | 32 | 0.7733 | 0.8447 | 0.8074 | 8/0/24/0 |
| 0.5929 | 169 | 37 | 37 | 0.8204 | 0.8204 | 0.8204 | 10/0/27/0 |

## Lectura

**El baseline se reproduce exactamente**: NMS 0.10 con prob 0.5929 da 37 FN, 37 FP, F1 0.8204 — idéntico al análisis previo de 37 FN. La comparación es válida.

**Bajar el umbral no ayuda.** De 0.5929 a 0.3: FN 37 → 13 (se recuperan 24 copas), pero FP 37 → 121 (se inventan 84). F1 cae de 0.8204 a 0.7423. El mejor F1 del barrido es justamente el baseline (0.8204).

**El borde persiste.** Con el umbral más bajo, de 13 FN todavía 8 cruzan el borde (2 vértices) y 5 son interiores. En el baseline eran 27 y 10. Bajar el umbral no elimina el problema de borde.

**Conclusión:** el umbral de probabilidad no es la causa de las copas perdidas; el modelo ya está operando en su punto óptimo de F1. La mitigación tiene que ir por contexto/escala (padding en inferencia o reentrenamiento con copas de borde), no por bajar `prob_thresh`.

Gráfico: `data/stardist_prob_sweep/prob_sweep.png`.
