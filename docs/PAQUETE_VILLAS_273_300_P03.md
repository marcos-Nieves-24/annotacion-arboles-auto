# Paquete CVAT — Villas 273-300 con prob_thresh=0.30 (2026-11-19)

## Decisión

Se eligió `prob_thresh=0.30` para los frames 273-300. Según el usuario, es un umbral bueno para detectar los árboles faltantes, aunque **permisivo**: puede incluir árboles que no son del cultivo. Es un riesgo aceptado explícitamente.

## Contenido

| Archivo | Descripción |
|---|---|
| `annotations.json` | 312 polígonos, 28 frames (273-300), `label_id=8155070`, `crop_tree` |
| `manifest.json` | `PREPARED`, parámetros, conteo por frame, advertencia de permisividad |
| `overlay_contact.png` | lámina con los 28 overlays (polígonos en rojo sobre la imagen original) |
| `overlay_<frame>.png` | overlays individuales |
| `backup_annotations_before_upload.json` | estado del job al preparar el paquete (GET de solo lectura) |

Ubicación: `data/para_subir/villas_273_300_p03/`.

Origen de los polígonos: `data/stardist_villas_273_300_sweep/0.3000/` (28 JSON del barrido).

## Parámetros

- Modelo: `limon-stardist-v1` (pesos congelados).
- `prob_thresh=0.30`, `nms=0.02`.
- Mapeo frame→imagen: `data/kaggle-villas-273-300/frame_map.json`, sacado de `meta.json` y verificado por hash.
- Comparación: 312 polígonos contra 224 con el umbral por defecto 0.5929 (+88).

## Dry-run

```bash
python3 scripts/upload_villas_273_300_p03_cvat.py
```

Resultado verificado:

```
job=4307835  shapes actuales=5310  nuevas=312  total=5622
frames nuevas=28  rango=273-300
incremento por frame: {273: 15, 274: 10, ... 300: 14}
DRY-RUN: no se escribió en CVAT
```

El payload no lleva `id` en las formas (CVAT rechaza PUT con ids). La subida fusiona: formas existentes + nuevas.

## Estado

**Preparado, no subido.** La escritura requiere `python3 scripts/upload_villas_273_300_p03_cvat.py --authorized` con autorización explícita.
