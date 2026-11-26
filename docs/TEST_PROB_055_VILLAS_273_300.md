# Test prob_thresh=0.55 — Villas frames 273-300

Mismo modelo y mapeo que la producción; único cambio: `prob_thresh=0.55` en vez de 0.5929.
NMS 0.02. Comparación contra los polígonos ya generados con 0.5929.

| Frame | baseline 0.5929 | test 0.55 | delta |
|---:|---:|---:|---:|
| 273 | 12 | 12 | +0 |
| 274 | 9 | 9 | +0 |
| 275 | 5 | 6 | +1 |
| 276 | 6 | 6 | +0 |
| 277 | 3 | 3 | +0 |
| 278 | 4 | 4 | +0 |
| 279 | 3 | 4 | +1 |
| 280 | 6 | 6 | +0 |
| 281 | 9 | 10 | +1 |
| 282 | 9 | 10 | +1 |
| 283 | 10 | 10 | +0 |
| 284 | 9 | 9 | +0 |
| 285 | 12 | 12 | +0 |
| 286 | 10 | 10 | +0 |
| 287 | 6 | 6 | +0 |
| 288 | 5 | 7 | +2 |
| 289 | 6 | 7 | +1 |
| 290 | 9 | 9 | +0 |
| 291 | 6 | 7 | +1 |
| 292 | 9 | 10 | +1 |
| 293 | 12 | 12 | +0 |
| 294 | 4 | 6 | +2 |
| 295 | 3 | 3 | +0 |
| 296 | 12 | 12 | +0 |
| 297 | 12 | 14 | +2 |
| 298 | 11 | 12 | +1 |
| 299 | 11 | 12 | +1 |
| 300 | 11 | 12 | +1 |

**Total:** baseline 224 → test 240 (+16).

Esto es sólo un conteo de polígonos, no una evaluación contra oro: no sabemos cuáles son aciertos y cuáles falsos positivos sin revisar los overlays.

Overlays: `data/stardist_villas_273_300_p55/overlays_p55.png`.
