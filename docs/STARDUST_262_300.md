# Anotación StarDist — frames 262-300 de Villas (2026-11-19)

Extensión del pipeline `limon-stardist-261-270` a un rango mayor. Se ejecuta **solo en Kaggle T4**: el venv local con Python 3.14 no compila las extensiones de `stardist`/`csbdeep`.

## 1. Mapeo frame → imagen

Regla verificada en el manifest del frame 262: `frame N = imagen N+510` (`00772` → frame 262, `00771` → frame 261).

| Imágenes | Frames | Disponibilidad |
|---|---|---|
| `00772`–`00789` | 262-279 | ✅ `data/kaggle-villas-261-270/images/` y `data/r1/images/villas/` |
| `00790`–`00804` | 280-294 | ❌ **no existen en el repo ni en Kaggle** |
| `00805`–`00810` | 295-300 | ✅ `data/r1/images/villas/` |

Se procesan las **24 imágenes disponibles** (opción A acordada con el usuario). Los 15 frames faltantes se incorporan en una segunda pasada cuando se consigan las imágenes.

## 2. Artefactos creados

| Elemento | Valor |
|---|---|
| Dataset de entrada | `marcosalvarez234/villas-262-300` (24 PNG, creado desde `data/kaggle-villas-262-300/`) |
| Kernel | `marcosalvarez234/limon-stardist-262-300` |
| Código | `kernels/stardist_262_300/kernel_source.py` |
| Metadata | `kernels/stardist_262_300/kernel-metadata.json` — incluye los 2 datasets de entrada |
| Pesos | dataset `marcosalvarez234/limon-stardist-v1-weights` (sin reentrenar) |
| Builder | `scripts/build_kernel_centroides.py` |

**Crítico:** el builder sobrescribe `kernel-metadata.json` con un solo `--dataset`, así que hay que volver a agregar `marcosalvarez234/villas-262-300` a la lista después de cada rebuild. Ya pasó en el kernel 261-270 y costó dos corridas fallidas.

## 3. Parámetros (idénticos al pipeline original)

- Pesos: `weights_best.h5` + `config.json` + `thresholds.json`, `StarDist2D(None, name='limon_stardist_v1')`.
- `prob_thresh` leído de `model.thresholds` (0.5929).
- Normalización: `csbdeep.utils.normalize(img, 1, 99.8, axis=(0,1))`.
- Barrido NMS: `0.02`, `0.05`, `0.10` → overlays `nms{N}_{imagen}.png`.
- Export CVAT a NMS `0.02`: contorno externo → `cv2.approxPolyDP(0.005·perímetro)`.
- Etiqueta `crop_tree`, `label_id=8155070`.

## 4. Salidas esperadas

En `data/stardist_262_300/` tras `kaggle kernels output`:

- `stardist_262_300_report.json` — máscaras, componentes y pares con solape por frame y por NMS.
- `nms*.png` — overlays por color (24 imágenes × 3 NMS).
- `cvat_<frame>_stardist.json` — poligonos por frame (24 archivos).
- `cvat_262_300_stardist.json` — shapes de todos los frames en un solo archivo.
- `manifest_262_300.json` — mapeo frame/imagen, conteos y parámetros.

## 5. CVAT

**No se sube nada.** Los paquetes quedan listos en disco. Toda escritura a CVAT requiere autorización explícita por lote. Nota: los scripts antiguos `experimentos/cascada/scripts/upload_26{1,2}_cvat.py` leen `ROOT/scripts/ronda1.py`, que después de la higiene quedó en `experimentos/yolo/scripts/ronda1.py`; hay que corregir esa ruta antes de cualquier subida.

### Paquete preparado para frames 263-300

- `data/para_subir/villas_263_300/` — 196 polígonos en 23 frames (263-279 y 295-300). Faltan 280-294.
- `annotations.json`, `manifest.json`, `overlay_contact.png` (lámina con los 23 overlays NMS 0.02) y `backup_annotations_before_upload.json` (GET de solo lectura al estado actual del job).
- El job ya tiene 4970 formas; los frames 263-300 tienen entre 2 y 8 formas cada uno. La subida **fusiona**: `PUT` con las formas existentes sin `id` + las nuevas.
- Script: `scripts/upload_263_300_cvat.py`. Por defecto hace **dry-run** (sólo GET + resumen). La escritura requiere `python3 scripts/upload_263_300_cvat.py --authorized`, y ese flag sólo se usa con autorización explícita del lote.
- El token se lee de `experimentos/yolo/scripts/ronda1.py`, ruta correcta después de la higiene.
- Dry-run verificado: 4970 existentes + 196 nuevas = 5166 a enviar.
- **Subida autorizada y ejecutada 2026-11-19**: `PUT` HTTP 200, 196 shapes enviadas, total 4970 → 5166. Verificación con GET posterior: **23/23 frames con el incremento exacto esperado**, sin desajustes. Ejemplos: frame 263 4→15 (+11), frame 300 7→19 (+12). `manifest.json` en `UPLOADED`.

## 6. Resultados (kernel COMPLETE, 2026-11-19)

Salidas en `data/stardist_262_300/` (41 MB, 103 archivos). Sin errores en el log.

| NMS | Máscaras | Componentes conexos | Pares con solape |
|---|---:|---:|---:|
| 0.02 | 212 | 176 | **0** |
| 0.05 | 224 | 169 | **0** |
| 0.10 | 240 | 163 | **0** |

- 24 overlays por NMS (72 en total) + 24 JSON de CVAT por frame + 1 combinado (`cvat_262_300_stardist.json`, 212 shapes).
- Frames cubiertos: 262-279 y 295-300. Faltan 280-294.
- Etiqueta `crop_tree`, `label_id=8155070`, `prob_thresh=0.5929`.

### Verificación de reproducibilidad

Los 16 polígonos del frame 262 generados en esta corrida son **idénticos** a los del pipeline original (`data/stardist_261_270/cvat_262_stardist.json`), confirmando que el cambio de `imgs[:10]` a las 24 imágenes no alteró el comportamiento.

### Estado de CVAT

**No se subió nada.** Los paquetes quedan listos en disco. Toda escritura a CVAT requiere autorización explícita por lote.
