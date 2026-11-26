# Anotación automática de copas de árboles

Pipeline para anotar automáticamente copas individuales de limón en imágenes RGB de dron (tiles de 512×512 px). Las copas se tocan entre sí, lo que hace que los métodos clásicos fallen. Combina un modelo de centroides (v1/v2/v3) con StarDist y sube los resultados a CVAT por lotes.

Solo se versiona código y documentación. Datos, modelos y outputs pesados quedan fuera del repo (ver `.gitignore`).

## Estructura

```
scripts/        # inferencia, paquetes CVAT, análisis
kernels/        # notebooks de Kaggle (StarDist + centroides) + metadata
notebooks/      # exploración local
docs/           # pipeline completo, decisiones, métricas, artículos
data/           # gitignored: imágenes, JSON de CVAT, modelos
odd/            # tareas de seguimiento
```

## Requisitos

- Python 3.10+
- GPU recomendada para entrenamiento (Kaggle T4). Local solo inferencia liviana.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cómo correr el pipeline

1. **Entrenamiento** (Kaggle): los kernels están en `kernels/` (StarDist y centroides).
2. **Inferencia local**:
   ```bash
   python scripts/infer_centroides_single.py
   ```
   Los kernels de StarDist exportados se usan desde los notebooks.
3. **Preparar paquetes para CVAT**:
   ```bash
   python scripts/prepare_upload_263_300.py
   # o las variantes magdala/villas
   ```
   Genera un JSON por frame en `data/para_subir/`.
4. **Subir a CVAT** (solo con autorización explícita por lote):
   ```bash
   export CVAT_TOKEN="tu-token"
   python scripts/upload_263_300_cvat.py
   ```
5. **Análisis**:
   ```bash
   python scripts/analyze_prob_sweep.py
   python scripts/render_confusion.py
   ```

El detalle completo del pipeline y las decisiones está en `docs/PIPELINE_COMPLETO.md`.

## Restricciones del proyecto

- Todo el entrenamiento corre en Kaggle T4. En local solo inferencia liviana, en foreground, con timeout.
- CVAT es solo lectura salvo autorización explícita por lote.
- Las credenciales no se commitean. El token de CVAT se lee de la variable de entorno `CVAT_TOKEN`.
- Solo RGB: sin LiDAR, multiespectral ni cámara de altura.

## Licencia

Pendiente.