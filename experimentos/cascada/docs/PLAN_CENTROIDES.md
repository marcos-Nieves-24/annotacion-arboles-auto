# Plan — Generador de propuestas por centroides (Frente B)

Fecha: 2026-11-18. Alcance: **diseño**. No entrena ni escribe en CVAT todavía.
Restricción del usuario (2026-11-18): **todo entrenamiento se hace en Kaggle**; en este equipo solo se preparan targets, se revisan resultados y se evalúan artefactos. No se dejan procesos abiertos que consuman swap.

## 1. Qué es y qué NO es

**Es** un generador de propuestas: dado un tile RGB 512×512, devuelve una lista de candidatos `(cx, cy, r)` — centro y radio equivalente de cada copa. Su trabajo es **encontrar copas que el detector actual pierde**, no producir la máscara final.

**No es** el estimador de conteo de `PLAN_234` (que dice *cuántas* pero no *dónde*), ni un reemplazo de YOLO11n-seg, ni un clasificador. Es la etapa que falta en la cascada: la que propone sobre-completamente para que el refinador y el validador puedan trabajar.

Rol en el sistema:

```
centroides (recall) ─┐
YOLO T1b conf 0.10-0.25 ─┼─→ unión de propuestas ─→ refinamiento (SAM/Cellpose)
FG-TreeSeg zero-shot ────┘                              ↓
                                      validador calibrado → aceptar / revisión / rechazar
```

## 2. Por qué puede recuperar lo que YOLO no ve

- Medimos **41% del oro de train sin detección** (IoU ≥ 0.25) con T1b; el punto operativo tiene R 0.653.
- En frames con 0 det a conf 0.25, había detecciones latentes a conf 0.01–0.24 en 20 de 21 frames, pero bajar el umbral traía maleza, no copas.
- Un heatmap de centros tiene un **inductive bias distinto**: no necesita que el modelo reconozca la forma completa de la copa para disparar; le basta el pico de textura en el centro. Donde YOLO requiere una máscara coherente, acá alcanza con un punto.
- La literatura respalda que puntos son supervisión suficiente: PointSAM (TGRS 2025) y TreePseCo (citado por FG-TreeSeg).

## 3. Etiquetas: gratis, ya están en el oro

Medido sobre `data/gold_v2/coco.json` (405 imágenes, 4602 anotaciones):

| Estadística | Valor |
|---|---|
| Radio equivalente `r = √(area/π)` — mediana / p10 / p90 | **51.4 / 25.1 / 70.3 px** |
| Villas (n=2419) — r mediana / p10 / p90 | 58.5 / 28.2 / 73.2 px |
| Magdala (n=2183) — r mediana / p10 / p90 | 45.4 / 22.4 / 61.7 px |
| Distancia centro a vecino más cercano — mediana / p10 / p5 | **103.6 / 73.3 / 64.0 px** |
| `dist_NN / (r_i + r_j)` — mediana / p10 | 0.99 / 0.81 |
| Splits | split_A 223 train / 162 val; split_B invertido; dev A=22, B=16; holdout=20 |

**Generación del target** (`scripts/build_centroid_targets.py`):

1. Centroide de cada polígono por **shoelace** sobre `segmentation` (no `bbox`: en `gold_v2` el campo `bbox` está en 0 para todas las anotaciones — trampa confirmada).
2. Radio `r = √(area/π)` en px. Escalar con GSD sólo si se reporta en metros.
3. Heatmap a **stride 4** (128×128): pico en `(cx/4, cy/4)`, Gaussian con `σ_full = clamp(0.25·r, 2, 16) px` → `σ_heat = σ_full/4`.
4. Solapamiento de Gaussians por **máximo elemento a elemento**, nunca suma (evita saturación en huertos densos).
5. Mapa de radio supervisado **solo en el píxel del centro** (y opcionalmente en el núcleo `H > 0.5`); el resto se enmascara como *ignore*.
6. Regla de borde: los centroides caen todos dentro del tile (verificado: 0 fuera), pero las copas recortadas tienen `r` subestimado. Marcarlas con `weight=0.5` en la pérdida de radio.
7. `iscrowd` (18% de las anotaciones) se entrena igual: el centroide del polígono sigue siendo válido.

## 4. Modelo

**CenterNet-lite**: backbone ResNet-18 pretrained ImageNet + cuello FPN/deconv hasta **stride 4**, dos cabezas 1×1:

| Head | Salida | Activación | Pérdida |
|---|---|---|---|
| Heatmap | 128×128×1 | sigmoid | Focal loss (α=2, β=4, CenterNet) |
| Radio | 128×128×1 | softplus | SmoothL1 sobre píxeles con pico, λ=1.0 |
| Offset (opcional) | 128×128×2 | tanh | SmoothL1, λ=0.1 |

- ¿Por qué ResNet-18 y no un ViT? 11 M parámetros, ya lo usamos en el clasificador de textura, corre en T4 y en CPU. El objetivo es recall de propuestas, no estado del arte.
- ¿Por qué stride 4 y no 8? La cuantización es ±2 px. Con r ≥ 20 px es despreciable; a stride 8 sería ±4 px y las copas chicas sufrirían.
- Se descarta copy-paste de textura: ya aprendimos con Rossi que genera atajos.

## 5. Hiperparámetros (punto de partida)

| Parámetro | Valor | Comentario |
|---|---|---|
| Input | 512×512×3, norm ImageNet | sin CLAHE (se prueba aparte) |
| Backbone | resnet18, ImageNet, stride 4 | FPN 64 canales |
| Optimizer | AdamW, lr 3e-4, wd 1e-4 | cosine decay |
| Batch | 16 en Kaggle T4 (sin entrenamiento local) | 512² × 16 ≈ 4 M px |
| Épocas | 80, patience 10 | early stopping en pérdida de val |
| Augment | flip H/V, rot ±15°, HSV ±0.015, escala 0.9–1.1 | si hay escala, escalar `r` igual |
| Focal α / β | 2 / 4 | default CenterNet |
| λ_radio | 1.0 | subir a 5 si la pérdida de radio no baja |
| Seed | 7 | reproducible |
| Decode top-K | 40 | máximo observado 33–35 copas/frame |
| Umbral de pico | 0.15 | barrir 0.10 / 0.15 / 0.20 |
| NMS por distancia | `0.7·(r_i + r_j)` | calibrar contra p10 real = 73 px; bajar a 0.6 si se pierden pares |
| Salida | `(cx, cy, r, score)` × 4 para coordenadas | — |

## 5b. Infraestructura: todo entrena en Kaggle

- **Entrenamiento: solo Kaggle T4.** Kernel nuevo `limon-centroides-v1`, con `accelerator: T4`, inspirado en `limon-fase2-yolo`. Ningún entrenamiento en este equipo: ni `.venv-verify`, ni CPU, ni GPU local.
- **Datos:** montar el dataset privado `marcosalvarez234/limon-gold-v2` ya existente (405 imágenes + 397 labels + coco/splits/dev/holdout/quarantine, status *ready*). No hace falta subir nada. Cualquier subida nueva requiere autorización explícita.
- **Credencial:** se pasa por variable de entorno de la sesión; nunca se escribe en `docs/`, `scripts/` ni en el código del kernel. Si aparece en el chat, no se repite.
- **Local solo:** E0 (generar targets y visualizar 20 frames), revisión de reportes y evaluación final sobre artefactos ya descargados. Comandos `foreground`, con `timeout` y sin `background: true`.

### Regla de procesos (no dejar swap consumido)

1. Antes de cada paso local: `free -h` y `pgrep -af 'python|jupyter|kaggle|torch'`. Si aparece algo nuestro colgado, se identifica por PID y se cierra antes de seguir.
2. Todo comando local lleva `timeout` explícito. Cero servidores, cero notebooks persistentes, cero `fo.launch_app(...)` con `wait()` abandonado.
3. Después de cada paso: verificar que no quede PID nuestro. El kernel de Kaggle se apaga solo al terminar; no hay proceso local que sobreviva.
4. Si un proceso no termina, se reporta el PID al usuario y se pide confirmación antes de matarlo. Nunca se mata un proceso de Brave o LibreOffice del usuario.
5. Artefactos (pesos, reporte JSON, overlays) se descargan al terminar y el kernel se detiene. Ninguna descarga queda corriendo en background.

## 6. Inferencia y cascada

1. Heatmap → máximos locales (maxpool 3×3) → top-K con score > umbral.
2. Radio por pico → NMS por distancia (no IoU: son puntos, no cajas).
3. Cada `(cx, cy, r)` se convierte en **box prompt** `[cx−r, cy−r, cx+r, cy+r]` para SAM/Cellpose.
4. Refinamiento: máscara candidata dentro de ese box.
5. Validador calibrado en oro: banda de radio por finca, solape con vecino ya aceptado, y conteo esperado como **alarma blanda**.
6. Rechazos → cola de revisión humana. **Nunca borrado silencioso.**

## 7. Métricas y gates

| Gate | Métrica | Criterio |
|---|---|---|
| E0 — targets | Visualizar 20 frames con Gaussians sobre la imagen | picos sobre el centro visible de la copa, no sobre el hueco |
| E1 — propuesta | **Recall de propuesta** en dev A/B y 9 frames auditados: fracción de centroides oro con un pico a ≤ `max(12 px, 0.25·r)` | ≥ 0.85 en auditados |
| E2 — integración | Instance F1 (IoU ≥ 0.5, one-to-one) en dev y holdout vs baseline 0.728 | ≥ 0.748 (es decir +0.02) |
| E3 — costo | FP nuevos y minutos de revisión humana | FP ≤ +2 pp, tiempo de anotación ≤ 0.5× el actual |

Los números de E1/E2 se producen **dentro del kernel de Kaggle** y se bajan como JSON; en local solo se leen, no se recalculan.

Reglas de datos: entrenar **solo** con el split `train` correspondiente; dev A/B y los 9 frames auditados son evaluación; **holdout 20 intacto**. Si un frame auditado cae en train, se excluye del entrenamiento (ya pasó con Frente A).

## 8. Riesgos y mitigaciones

- **Picos fusionados en coronas que se tocan**: Gaussian por max + NMS por distancia calibrado con el p10 real (73 px). Si se pierden pares, bajar el factor de 0.7 a 0.6.
- **Radio subestimado en copas de borde**: weight 0.5 en la pérdida; el validador no debe usar esas copas para calibrar la banda de área.
- **Maleza**: el heatmap puede disparar en arbustos (mismo problema del clasificador de textura, FP 41%). Mitigación: el validador con banda de radio por finca y la revisión humana; no se acepta en forma ciega.
- **Poco datos** (223 frames): augment fuerte y early stopping; si sobreajusta, bajar a ResNet-18 sin FPN o usar `mit-b0`.
- **Costo**: es un modelo chico; el cuello va a ser el refinamiento con SAM, no el heatmap.

## 9. Extensión posterior (no incluida en el primer experimento)

Prior estructural de huerto: segmentar **filas** de plantación y colocar candidatos a lo largo de cada fila con espaciado derivado del radio mediano. Es una señal que los bosques no tienen y que la literatura de cítricos solapados ya usa. Solo si E1 muestra que el heatmap solo no alcanza.

## 10. Log

- 2026-11-18: usuario pidió elaborar la idea de centroides con parámetros, modelo e hiperparámetros.
- 2026-11-18: usuario estableció que todo entrenamiento va a Kaggle y que no se dejan procesos abiertos consumiendo swap. Agregada sección 5b con kernel `limon-centroides-v1`, credencial por entorno y regla de procesos.
- 2026-11-18: Stats extraídos de `data/gold_v2/coco.json` con shoelace sobre polígonos: r mediana 51.4 px, distancia a vecino mediana 103.6 px, p10 73.3 px. Descubierto: `bbox` en cero en gold_v2 → usar `segmentation`.
