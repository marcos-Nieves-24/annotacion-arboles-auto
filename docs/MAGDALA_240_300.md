# Pipeline StarDist — Magdala frames 240-300 (2026-11-19)

Extensión del pipeline StarDist a Magdala, con el mapeo de frames verificado contra la fuente autoritativa.

## 1. Mapeo frame → imagen

Fuente: `data/raw/magdala/meta.json`, la misma que usa `experimentos/yolo/scripts/ronda1.py` para descargar del job. **No se usó aritmética** `frame = imagen - N`.

- Frame 240 → `Limon_Magdala_01267.png`
- Frame 300 → `Limon_Magdala_01366.png`
- 61 imágenes, todas disponibles en `data/r1/images/magdala/`.
- Verificación final: 61/61 imágenes del manifest coinciden con `meta.json`, 0 desajustes.

## 2. Artefactos

| Elemento | Valor |
|---|---|
| Dataset de entrada | `marcosalvarez234/magdala-240-300` (61 PNG + `frame_map.json`) |
| Kernel | `marcosalvarez234/limon-stardist-magdala-240-300` |
| Código | `kernels/stardist_magdala_240_300/kernel_source.py` |
| Pesos | `marcosalvarez234/limon-stardist-v1-weights` (sin reentrenar) |
| Salidas | `data/stardist_magdala_240_300/` (61 JSON + 61 overlays) |
| Reporte | `stardist_resto_report.json` (nombre heredado del kernel parametrizado) |
| Paquete CVAT | `data/para_subir/magdala_240_300/` |

El kernel reutiliza el esquema de `stardist_resto`: lee `frame_map.json` del dataset en vez de inferir el frame del nombre de archivo.

## 3. Parámetros

Idénticos al pipeline StarDist original: `prob_thresh=0.5929`, `normalize(1, 99.8, axis=(0,1))`, NMS 0.02, `cv2.approxPolyDP(0.005·perímetro)`, etiqueta `crop_tree`, `label_id=8155070`.

Resultado: **714 polígonos en 61 frames**. Ejemplos: frame 240 → 14 polígonos / 9 componentes; frame 250 → 7/6; frame 300 → 6/6.

## 4. CVAT

Job **4307818** (Magdala). El job ya tiene 10060 formas; en los frames 240-300 hay entre 2 y 23 formas por frame. La subida **fusiona** (`PUT` con existentes sin `id` + nuevas).

- `scripts/prepare_upload_magdala_240_300.py` construye el paquete.
- `scripts/upload_magdala_240_300_cvat.py` hace **dry-run por defecto**; requiere `--authorized`.
- Backup del estado previo: `data/para_subir/magdala_240_300/backup_annotations_before_upload.json`.

Dry-run verificado: 10060 existentes + 714 nuevas = 10774 a enviar, 61 frames (240-300).

## 6. Centroides (resultado paralelo)

Se corrió el modelo de centroides v1 congelado sobre las mismas 61 imágenes, en local (CPU, foreground, sin procesos residuales), con `scripts/infer_centroides_single.py`.

- Decodificación idéntica a la cascada: `thr=0.5`, `topk=40`, `dist_factor=0.7`, radio fijo de Magdala 45.4 px.
- **957 centroides** en 61 frames (rango por frame: 2-22), contra 714 polígonos de StarDist.
- Salidas: `data/centroides_magdala_240_300/centroides_240_300.json` (consolidado), `centroides_240_300/<frame>/centroides.json` + `centroides_overlay.png` por frame, y `contact_sheet.png`.
- El mapeo frame→imagen se tomó del mismo `frame_map.json` del dataset, verificado contra `meta.json`.
- Esto **no se subió a CVAT**; es un resultado paralelo en disco.

## 5. Estado

**Subida autorizada y ejecutada 2026-11-19.** Revisión previa del job: sin cambios desde el backup (10060 formas). `PUT` HTTP 200, 714 shapes enviadas, total 10060 → 10774. Verificación con GET posterior: **61/61 frames con el incremento exacto**, sin desajustes. Ejemplos: frame 240 23→37 (+14), frame 250 14→21 (+7), frame 300 8→14 (+6). `manifest.json` en `UPLOADED`.
