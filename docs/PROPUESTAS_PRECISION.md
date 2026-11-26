# Propuestas de alta precisión para copas sin intersecciones — 2026-11-19

Alcance: revisión bibliográfica + evidencia propia. **No se implementó ninguna.** Las opciones se ordenan por precisión esperada y por compatibilidad con lo que ya tenemos.

## Lo que ya sabemos (evidencia propia)

- v1.2/v4 detecta bien (F1 dev 0.9332, holdout 0.8998) pero deja 39 pares de máscaras con IoU > 0.05 y 13 con IoU > 0.15 en 10 frames.
- El priority painting elimina el solape de píxeles pero **no separa las copas**: 5 componentes conexos contenían varias instancias en `00772`, 6 en `00775`, y una copa se partió en 26 componentes en `00779`. Queda descartado como solución de separación.
- Voronoi / watershed sobre distancia separan pero cortan lóbulos (área ratio 0.64).
- opt3 conserva forma (área 0.89) pero pierde 7 copas.
- **Nuevo test decisivo**: las copas del oro son casi perfectamente star-convex. Reconstruyendo 576 polígonos con 36 rayos desde el centroide: IoU p10 0.958, mediana **0.981**, p90 0.987; 99.8% por encima de 0.8. El centroide cae dentro del polígono en el 100% de los casos.

## Hallazgos de literatura

| Trabajo | Resultado relevante | Limitación para nosotros |
|---|---|---|
| **Tong & Zhang, RSE 2025** — StarDist para ITC en RGB aéreo | >92% accuracy, F1 0.88, **+6% sobre Mask R-CNN**, R² de área >0.85, entrena con pocos datos | Bosque mixto, no huerto; requiere polígonos star-convex |
| **ISPRS 2026 — StarDistSparse** (dehesa, encinas) | F1 0.85 con IoU 0.5 sobre 1200 árboles; el **umbral de overlap/NMS se puede bajar hasta casi 0 si no se quiere ninguna superposición** | En cobertura densa aumenta la omisión |
| **ISPRS 2025 — HTC + CB-ResNet50** (concurso ITC) | Mejor modelo: AP50 40.98, AP75 21.25; SAM solo AP50 8.36 | AP absoluto modesto; datos de competencia mucho más grandes (7267 imgs) |
| **NeurIPS 2025 — RSPrompter / BalSAM** | Aprender prompts para SAM supera a Mask R-CNN y a Mask2Former en 3 datasets; SAM out-of-the-box es subóptimo y fusiona copas que se tocan | Requiere entrenar un prompter ligero; DSM ayuda pero no lo tenemos |
| **FG-TreeSeg, 2026** | Flujo de Cellpose-SAM separa copas que se tocan por convergencia, sin anotaciones de instancia; mAP50 67.31 vs 69.05 supervisado | Needs semantic mask previa y Cellpose-SAM |
| **CVPRW 2025 — SAM-M2F** (huertos frutales) | Extensión de etiquetas con SAM 2 + Mask2Former: AP 64.1 vs 54.3 de Mask2Former solo | Pensado para video/scans verticales, no tiles aislados |
| **ScienceDirect 2025 — SAM 2 + LiDAR** | Box prompts precisos dan F1 de píxel 78–87% | LiDAR no disponible; confirma que el box importa más que SAM |

## Propuestas

### Propuesta A — StarDist sobre el oro (alta precisión, mejor calce)

**Idea**: reemplazar toda la cascada por un modelo StarDist entrenado con nuestros 405 frames de oro. StarDist predice, por píxel, probabilidad de objeto y distancias a los bordes en 36 rayos; genera muchos polígonos candidatos y usa NMS. Si el umbral de overlap/NMS se pone cerca de 0, **dos copas no pueden solaparse por construcción**.

**Por qué es la favorita**:
- Nuestro test dice que las copas son star-convex con IoU 0.981 — justo la suposición del modelo.
- Literatura reporta F1 0.88–0.92 y superioridad sobre Mask R-CNN en exactamente este problema.
- Resuelve la intersección en el modelo, no con postproceso: no hay cesión de píxeles ni lóbulos cortados.
- Entrena con pocos datos y corre en Kaggle T4.

**Riesgos**: el oro tiene 18% de `iscrowd` (polígonos que tocan borde o son fusiones) y 217 anotaciones en cuarentena; habría que filtrar y convertir a label image con ID único por copa. En copas muy juntas puede aumentar la omisión.

**Validación**: F1/IoU en holdout vs v1.2, y conteo de pares con IoU > 0.05 (objetivo 0).

### Propuesta B — FG-TreeSeg (separación por flujo, sin reentrenamiento)

**Idea**: usar el flujo de Cellpose-SAM con una máscara semántica de dosel como restricción. Cada píxel corre hacia un sumidero (centro de copa); la frontera emerge donde los flujos divergen, sin dibujar bordes ni partir geometrías.

**Por qué**: está diseñado exactamente para "copas que se tocan y no se ven separadas", que es nuestro caso sin surco. Es training-free y transferible a dominios nuevos.

**Qué necesitamos**: una máscara de dosel. Podemos construirla con la unión de máscaras SAM aceptadas o con un umbral de vegetación — ya tenemos ambos. Cellpose-SAM no está instalado en local pero se puede instalar en el kernel (el repo del paper lo indica).

**Riesgos**: dependencia nueva pesada; el `diameter` de Cellpose debe calibrarse con nuestro radio mediano (58.5 px Villas / 45.4 Magdala / 36 Yulima). Puede sobre-segmentar maleza si la máscara semántica no es buena.

**Validación**: misma métrica + revisión visual de los 10 frames problématicos.

### Propuesta C — Mask2Former / HTC supervisado con etiquetas del oro

**Idea**: fine-tunear un modelo de instancias (HTC + CB-ResNet50 es el mejor del concurso ISPRS; Mask2Former con Swin también evaluado) sobre nuestros 405 frames, limpiando `iscrowd` y cuarentena.

**Por qué**: es la línea base más fuerte en literatura cuando hay datos etiquetados suficientes. Nuestro oro es suficiente para un fine-tune, y SAM-M2F sugiere que se puede ampliar con propagación de etiquetas.

**Riesgos**: los papers muestran que Mask2Former **no supera** a Mask R-CNN en algunos datasets forestales; AP50 del concurso es 0.41, más bajo que nuestro F1 actual. Requiere más datos o augmentations agresivas. El oro tiene copas fusionadas que ensucian la supervisión. Es la opción más cara y no está claro que gane.

### Propuesta D — RSPrompter / BalSAM (aprender a prompear SAM)

**Idea**: entrenar un prompter ligero que genere prompts óptimos para SAM usando embeddings de imagen, manteniendo SAM congelado. NeurIPS 2025 reporta que supera a Mask R-CNN y Mask2Former.

**Por qué**: ataca la causa de nuestros errores — el box prompt con radio fijo y SAM eligiendo máscaras que invaden vecinos.

**Riesgos**: requiere implementar un prompt learner; probablemente necesite DSM para alcanzar los números del paper, que no tenemos. Alto costo de implementación.

## Recomendación

**Orden: A → B → C → D.**

1. Empezar por **StarDist**: nuestro test de star-convexidad (0.981 IoU) es una validación directa de su suposición, y la literatura reporta que resuelve copas que se tocan con NMS. Es la opción con mayor probabilidad de dar copas individuales, sin intersección y con buena forma.
2. Si StarDist falla en las copas más densas, agregar **FG-TreeSeg** como separador de flujo sobre las máscaras resultantes.
3. Los modelos supervisados grandes quedan como respaldo: hoy nuestro F1 es más alto que los AP publicados, y el costo de entrenarlos no está justificado sin antes agotar A y B.

## Lo que NO resolvería ninguna opción

El caso de una copa literalmente **encima** de otra en 3D. Sin DSM/LiDAR, ningún método 2D puede separarlas de forma confiable. Es una limitación de datos, no de modelo.

## Artefactos de esta revisión

- `docs/PROPUESTAS_PRECISION.md` (este archivo).
- Test de star-convexidad: 576 polígonos del oro, 36 rayos, mediana 0.981.
- Literatura citada con URLs en la tabla.
