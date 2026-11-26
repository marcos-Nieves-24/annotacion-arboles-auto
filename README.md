# Anotación automática de copas de árboles (pipeline)

Pipeline de anotación automática de copas individuales de limón en imágenes RGB de dron (tiles 512×512), con copas que se tocan. Combina un modelo de centroides (v1/v2/v3) y StarDist, y sube resultados a CVAT por lotes.

> Este repo versiona **código y documentación**. Los datos, modelos y outputs pesados viven fuera del repo (ver `.gitignore`).

## Estructura

```
scripts/        # inferencia, preparación de paquetes CVAT, análisis
kernels/        # notebooks Kaggle (StarDist + centroides) + metadata
notebooks/      # exploración local
docs/           # pipeline completo, decisiones, métricas, artículos
data/           # (gitignored) imágenes, CVAT JSON, modelos
odd/            # tareas de seguimiento (ODD)
```

## Requisitos

- Python 3.10+
- GPU recomendada para entrenamiento (Kaggle T4); local solo inferencia liviana.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Flujo general

1. **Entrenamiento** (Kaggle): kernels en `kernels/` (StarDist y centroides).
2. **Inferencia local**: `scripts/infer_centroides_single.py` y kernels de StarDist exportados.
3. **Preparación de paquetes CVAT**: `scripts/prepare_upload_*.py` generan JSON por frame.
4. **Subida a CVAT**: `scripts/upload_*_cvat.py` (solo con autorización explícita por lote).
5. **Análisis**: `scripts/analyze_*.py`, `scripts/render_confusion.py`.

Ver `docs/PIPELINE_COMPLETO.md` para el detalle completo y decisiones.

## Restricciones del proyecto

- Todo entrenamiento en Kaggle T4; local solo inferencia liviana, `foreground`, con `timeout`.
- CVAT solo lectura salvo autorización explícita por lote.
- Sin credenciales en el repo; la key se lee desde `scripts/ronda1.py` (no versionado).
- Sin LiDAR, multiespectral ni cámara de altura: solo RGB.

## Licencia

Pendiente.