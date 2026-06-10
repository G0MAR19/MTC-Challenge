# CLAUDE.md — SMART Challenge 2026: IA para la Movilidad del Perú

> Este archivo es el documento de contexto principal para Claude Code.
> Contiene los objetivos, entregables, decisiones técnicas y estado actual del proyecto.

---

## Identidad del Proyecto

| Campo | Detalle |
|---|---|
| **Competencia** | SMART Challenge 2026: IA para la Movilidad del Perú |
| **Organizador** | Ministerio de Transportes y Comunicaciones (MTC) del Perú |
| **Plataforma** | Kaggle (`mtc-smart-challenge-ia-para-la-movilidad-del-peru`) |
| **Desarrollador** | Harold Victor Reyna Yangali |
| **Licencia del código** | GPL v3 (obligatorio por bases del concurso) |

---

## Problema a Resolver

Detectar y clasificar vehículos en intersecciones urbanas peruanas a partir de clips de video de **~5 segundos a 10 FPS (~50 frames por clip)**.

La detección se realiza mediante **Oriented Bounding Boxes (OBB)**: cajas delimitadoras con rotación, que capturan mejor la geometría de vehículos en tomas con perspectiva o aéreas.

### Clases objetivo (9 en total)

| ID | Clase | Notas |
|---|---|---|
| 1 | auto | Clase más frecuente |
| 2 | combi | |
| 3 | microbus | |
| 4 | minibus | |
| 5 | omnibus | |
| 6 | articulado | Clase rara |
| 7 | camion | |
| 8 | mototaxi | Clase rara |
| 9 | motocicleta | |

---

## Métrica Oficial

**Macro AP-rIoU @ [0.50 : 0.80]**

- Se calcula Average Precision con rotated IoU (rIoU) en umbrales 0.50, 0.55, …, 0.80, luego se promedia.
- El promedio es **macro**: cada clase pesa igual independientemente de su frecuencia.
- **Consecuencia crítica**: las clases raras (articulado, mototaxi, motocicleta) tienen el mismo peso que `auto`. El desbalance de clases es el problema más importante a resolver.

---

## Formato de Predicción (submission)

Cada fila del CSV de predicción corresponde a un frame. Si hay detecciones:

```
score;category_id;cx;cy;width;height;angle_deg[;score;category_id;...]
```

Si no hay objetos detectados en el frame:

```
none
```

Campos por detección: `score` (float, confianza), `category_id` (int 1–9), `cx cy` (centro del OBB en píxeles), `width height` (dimensiones en píxeles), `angle_deg` (ángulo de rotación en grados).

---

## Entregables Finales del Concurso

1. **Código fuente** — Repositorio completo bajo GPL v3.
2. **Memoria Descriptiva (Dossier Técnico)** — Documento técnico que explica la arquitectura, decisiones de diseño, resultados y análisis.
3. **Manual de Operación** — Instrucciones para reproducir el entorno, entrenar y ejecutar inferencia.
4. **Reporte CSV de Flujos Vehiculares** — Análisis de conteos y movimientos en intersecciones generado a partir de las predicciones del modelo.

---

## Arquitectura Técnica Decidida

### Modelo Principal

**YOLO con soporte nativo de OBB** (Ultralytics YOLOv8-OBB o YOLOv11-OBB).

- Soporte nativo de OBB: no requiere post-procesamiento adicional para la rotación.
- Alta velocidad de inferencia: viable para procesar 50 frames por clip.
- Alternativa evaluable: Rotated Faster R-CNN si se necesita mayor precisión en clases raras.

### Pipeline General

```
Dataset (clips de video)
    │
    ▼
Extracción de frames (cv2.VideoCapture @ 10 FPS)
    │
    ▼
Preprocesamiento + Data Augmentation OBB-aware
    │
    ▼
Entrenamiento YOLO-OBB (fine-tuning desde COCO/DOTAv1 pretraining)
    │
    ▼
Inferencia por frame
    │
    ▼
Post-procesamiento temporal (tracking entre frames del clip)
    │
    ▼
Generación de submission CSV
```

### Estrategias para el Desbalance de Clases

- **Weighted Random Sampler**: sobremuestreo de clases raras durante el entrenamiento.
- **Focal Loss**: penaliza más los ejemplos fáciles; favorece clases difíciles/raras.
- **Mixup / CutMix OBB-aware**: augmentation que preserva las coordenadas rotadas.
- **Oversampling explícito**: duplicar/triplicar clips que contengan clases raras (articulado, mototaxi, motocicleta).
- **Class-balanced batch sampling**: garantizar que cada batch contenga al menos K instancias de cada clase.

### Data Augmentation para OBB

Todas las transformaciones deben preservar la rotación de los OBB:

- Rotación aleatoria (0°–360°, actualiza `angle_deg`)
- Flip horizontal y vertical (actualiza signo del ángulo)
- Escala y recorte (actualiza `cx, cy, width, height`)
- Cambios de brillo/contraste/saturación (no afectan OBB)
- Perspectiva aleatoria ligera
- Librería recomendada: `albumentations` con soporte OBB (`BBoxParams(format='yolo_obb')`)

### Aprovechamiento Temporal (50 frames por clip)

- **Tracking intra-clip**: propagar detecciones entre frames usando ByteTrack o BoT-SORT.
- **Post-procesamiento temporal**: si un objeto es detectado con alta confianza en frame N, reforzar su detección en frames N±1 aunque la confianza sea baja.
- **Interpolación**: rellenar frames sin detección usando la trayectoria del tracker.
- **Score aggregation**: promediar scores de un mismo objeto tracking a través del clip.

---

## Estructura del Repositorio

```
MTC Challenge/
├── CLAUDE.md                  ← Este archivo (contexto para Claude Code)
├── README.md                  ← Descripción pública del proyecto
├── requirements.txt           ← Dependencias Python
├── MTC_Challengue.ipynb       ← Notebook principal (EDA + entrenamiento)
├── .venv/                     ← Entorno virtual (Python 3.13)
├── data/                      ← Dataset (no versionado en git)
│   ├── train/                 ← Frames o clips de entrenamiento
│   ├── test/                  ← Frames o clips de evaluación
│   └── train.csv              ← Anotaciones OBB de entrenamiento
├── models/                    ← Pesos entrenados (.pt)
├── notebooks/                 ← Experimentos adicionales / análisis
├── src/                       ← Código fuente modular
│   ├── convert_to_yolo_obb.py ← Convierte train.csv a labels YOLO-OBB (4 esquinas)
│   ├── make_split.py          ← Split train/val por video_id + estructura YOLO
│   ├── train.py               ← Script de entrenamiento
│   ├── inference.py           ← Inferencia sobre clips de video
│   ├── tracking.py            ← Post-procesamiento temporal / tracking
│   ├── evaluate.py            ← Cálculo de macro AP-rIoU
│   └── submission.py          ← Generación del CSV de submission
├── outputs/                   ← Predicciones y reportes generados
│   ├── submission.csv
│   └── vehicle_flow_report.csv
└── docs/                      ← Dossier técnico y manual de operación
    ├── dossier_tecnico.docx
    └── manual_operacion.docx
```

---

## Estado Actual del Proyecto

### ✅ Completado

- [x] Entorno virtual configurado (`.venv`, Python 3.13)
- [x] `requirements.txt` definido (numpy, pandas, torch, albumentations, shapely, etc.)
- [x] Autenticación con Kaggle via `kagglehub`
- [x] Dataset descargado en `/root/.cache/kagglehub/competitions/mtc-smart-challenge-ia-para-la-movilidad-del-peru`
- [x] Notebook base `MTC_Challengue.ipynb` creado con secciones 1 y 2

### 🔄 En Progreso

- [ ] EDA completo: distribución de clases, estadísticas OBB, visualización de frames
- [ ] Exploración de la estructura real del dataset (CSV columns, carpetas de imágenes/video)

### 📋 Pendiente

- [x] Pipeline de conversión al formato YOLO-OBB (`convert_to_yolo_obb.py` + `make_split.py`)
- [ ] Fine-tuning de YOLOv8-OBB / YOLOv11-OBB desde pesos DOTA preentrenados
  - Augmentations OBB-aware: por ahora vía built-ins de Ultralytics (ver `configs/train.yaml`)
- [ ] Implementar tracking intra-clip (ByteTrack o BoT-SORT)
- [ ] Implementar evaluación con Macro AP-rIoU
- [ ] Generación del CSV de submission
- [ ] Generar reporte de flujos vehiculares
- [ ] Redactar Dossier Técnico y Manual de Operación

---

## Convenciones de Código

- **Lenguaje**: Python 3.13
- **Estilo**: PEP 8 estricto; docstrings en español para funciones públicas
- **Modularidad**: cada componente del pipeline en su propio módulo en `src/`
- **Reproducibilidad**: fijar semillas (`random`, `numpy`, `torch`) en todos los scripts
- **Logging**: usar `logging` estándar, no `print`, en scripts de entrenamiento e inferencia
- **Tipado**: type hints en todas las funciones
- **Licencia**: GPL v3 — incluir header en cada archivo `.py`

### Header GPL v3 obligatorio para cada archivo `.py`

```python
# SMART Challenge 2026: IA para la Movilidad del Perú
# Copyright (C) 2026 Harold Victor Reyna Yangali
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
```

---

## Dependencias Clave

| Librería | Uso |
|---|---|
| `ultralytics` | YOLO-OBB: entrenamiento e inferencia |
| `torch` / `torchvision` | Backend de deep learning |
| `albumentations` | Data augmentation OBB-aware |
| `shapely` | Cálculo de rIoU entre polígonos rotados |
| `opencv-python` | Lectura de video, extracción de frames, visualización OBB |
| `pandas` / `numpy` | Manipulación de anotaciones y métricas |
| `kagglehub` | Descarga del dataset y submission |
| `scipy` | Utilidades numéricas |
| `tqdm` | Barras de progreso en training loops |
| `tensorboard` | Monitoreo de métricas de entrenamiento |

> **Nota**: `pandas-profiling` excluido por incompatibilidad con Python 3.13. Alternativa: `ydata-profiling`.

---

## Notas Técnicas Importantes

1. **Resolución de rutas de kagglehub**: la ruta del dataset no es fija entre sesiones. Siempre resolver dinámicamente con `os.walk` buscando `train.csv`, no hardcodear paths.

2. **Formato de ángulo**: verificar si el dataset usa grados o radianes, y si el ángulo es respecto al eje X o Y. Normalizar antes de cualquier operación.

3. **rIoU vs IoU**: la métrica usa *rotated* IoU. Para evaluación local, usar `shapely.geometry.Polygon` para calcular intersección entre polígonos rotados.

4. **DOTA pretraining**: los modelos YOLO-OBB de Ultralytics incluyen pesos preentrenados en DOTAv1 (detección aérea con OBB). Usar estos como punto de partida, no pesos COCO estándar.

5. **Inferencia en test**: el set de test tiene clips de video. La inferencia debe procesar frame a frame y agregar resultados respetando el formato de submission.
