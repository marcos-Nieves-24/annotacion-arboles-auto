# Corrección de mapeo frame → imagen (2026-11-19)

## Error

Se asumió `frame N = imagen N+510`. Eso es válido hasta el frame 279, pero **falso desde el 280**. La fuente correcta es `data/raw/villas/meta.json`, la misma que usó el pipeline original (`experimentos/yolo/scripts/ronda1.py`, función `dl`):

| Frame | Imagen real | Imagen que yo supuse |
|---:|---|---|
| 280 | `Limon_LasVillas_00805.png` | `00790` (no existe en el task) |
| 294 | `Limon_LasVillas_00819.png` | `00804` |
| 295 | `Limon_LasVillas_00821.png` | `00810` |
| 300 | `Limon_LasVillas_00826.png` | `00815` |

Verificado con descarga de solo lectura desde CVAT y comparación de hashes SHA256: el frame 280 bajado del job coincide byte a byte con `00805`, y el frame 295 con `00821`.

Por eso el "hueco de 15 imágenes" (`00790`–`00804`) era inexistente: esas imágenes nunca estuvieron en el task de CVAT.

## Impacto en la subida del 2026-11-19

- Frames **262-279**: correctos. No se tocan.
- Frame **262**: ya estaba subido con StarDist, correcto.
- Frame **261**: cascada v4, sin StarDist.
- Se subieron **55 polígonos a los frames 295-300** que corresponden a las imágenes `00805`–`00810` (frames reales **280-285**). Esas formas están mal ubicadas.

## Corrección preparada

- Dataset `marcosalvarez234/villas-resto`: 22 imágenes reales (261, 280-294, 295-300) + `frame_map.json`.
- Kernel `marcosalvarez234/limon-stardist-restantes`: usa el mapa real, no aritmética. Salidas en `data/stardist_resto/` (22 JSON, 194 polígonos).
- Script `scripts/correct_cvat_frames.py`:
  - Quita las 55 formas mal ubicadas (identificadas por `frame ∈ 295..300` y coincidencia exacta de `points` contra el paquete anterior).
  - Agrega las 194 formas corregidas con su frame real.
  - Dry-run por defecto; requiere `--authorized`.

Dry-run verificado sobre el estado actual del job (5162 formas — cuatro menos que los 5166 posteriores a la subida, por ediciones en CVAT):

```
a quitar:  55  {295:6, 296:9, 297:9, 298:10, 299:9, 300:12}
a agregar: 194 en 22 frames
total final: 5301
```

## Nota sobre el frame 261

El frame 261 ya tiene 18 formas de la cascada v4. La corrección **agrega** 12 de StarDist sin quitar las de cascada, manteniendo el criterio usado en los frames 262 y 263-300 (fusionar, no reemplazar). Si se quiere evitar duplicados, hay que quitar explícitamente las 12 formas de cascada antes de agregar.

## Estado

Corrección **ejecutada y verificada** (2026-11-19), con autorización explícita del usuario.

- Primer intento: HTTP 400 porque el payload conservaba el campo `id` en las formas existentes. CVAT exige formas sin `id` en `PUT`. Confirmado que no hubo mutación (5158 formas antes y después).
- Fix: quitar `id` de las formas conservadas y añadir assertion local que lo verifica antes de enviar.
- Segundo intento: **HTTP 200**, 55 quitadas + 194 agregadas, total 5158 → 5297.
- Verificación con GET posterior:
  - 0 formas mal ubicadas siguen presentes.
  - 22/22 frames con el conteo esperado (`backup` + nuevas del reporte corregido).
  - Ejemplos: frame 261 18→30 (+12 StarDist), frame 280 6→12 (+6), frame 295 5→8 (+3), frame 300 7→18 (+11).
- Backup del estado previo a la corrección: `data/stardist_resto/backup_annotations_before_fix.json`.
