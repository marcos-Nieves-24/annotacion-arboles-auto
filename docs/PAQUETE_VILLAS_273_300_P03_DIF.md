# Paquete CVAT — sólo polígonos nuevos (Villas 273-300, 2026-11-19)

El usuario pidió **no agregar los 312 polígonos** del umbral 0.30, sino únicamente los que aparecieron respecto del baseline 0.5929, para no duplicar lo que ya está en CVAT.

## Criterio de diferencia

Por frame, cada polígono de `prob=0.30` se compara contra todos los polígonos baseline del mismo frame. Se marca como **nuevo** si el IoU de las máscaras rasterizadas es **< 0.5** con todos ellos.

Fuentes del baseline:

- Frames 273-279 → `data/para_subir/villas_263_300/annotations.json`
- Frames 280-300 → `data/stardist_resto/cvat_<frame>_stardist.json`

## Resultado

| | Polígonos |
|---|---:|
| Baseline 0.5929 | 224 |
| Umbral 0.30 | 312 |
| **Nuevos a subir** | **88** |

La diferencia coincide exactamente con `312 − 224`, así que el matching no perdió ni inventó ningún caso. Incrementos por frame: de 1 a 6 polígonos (ver `manifest.json`). El paquete anterior (`villas_273_300_p03/`, 312 polígonos) queda descartado en favor de éste.

## Archivos

`data/para_subir/villas_273_300_p03_diff/`:

- `annotations.json` — 88 shapes, frames 273-300, `label_id=8155070`.
- `manifest.json` — `PREPARED`, criterio, fuentes y conteos.
- `overlay_contact.png` + `overlay_<frame>.png` — los polígonos nuevos en **verde** sobre la imagen.
- `backup_annotations_before_upload.json` — estado del job al preparar (5310 shapes).

Script: `scripts/prepare_upload_villas_273_300_p03_diff.py`.

## Dry-run

```
job=4307835  shapes actuales=5310  nuevas=88  total=5398
frames nuevas=28  rango=273-300
DRY-RUN: no se escribió en CVAT
```

Subida: `python3 scripts/upload_villas_273_300_p03_diff_cvat.py --authorized` (requiere autorización explícita).

## Estado final

**Subida autorizada y ejecutada 2026-11-19.** `PUT` HTTP 200, 88 shapes enviadas, total 5306 → 5394. Verificación con GET posterior: 28/28 frames con el incremento exacto, los 88 polígonos nuevos presentes una sola vez, sin duplicados. `manifest.json` en `UPLOADED`. Nota: el job tenía 5306 shapes al momento de la subida (el usuario sigue editando en CVAT); el script partió del estado fresco.

## Advertencia

Estos 88 polígonos vienen del umbral permisivo 0.30. Pueden incluir árboles que no son del cultivo, tal como aceptó el usuario.
