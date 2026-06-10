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

"""
Oversampling de clases raras mediante Repeat Factor Sampling (RFS).

Contexto
--------
La métrica oficial es *macro* AP-rIoU: cada clase pesa igual. Con un desbalance
severo (auto = 80%, articulado = 0.04%) el modelo casi no ve las clases raras y
la métrica macro se hunde. Ultralytics YOLO-OBB no admite pesos de clase, así
que la estrategia es **mostrar más veces** los frames que contienen clases raras.

Algoritmo (Repeat Factor Sampling, Gupta et al., "LVIS", CVPR 2019)
-------------------------------------------------------------------
1. Para cada clase c se calcula su frecuencia a nivel de imagen:
       f_c = (nº de imágenes que contienen c) / (nº total de imágenes)
2. Factor de repetición por clase:
       r_c = max(1, sqrt(t / f_c))
   donde `t` es un umbral. Clases más raras → r_c más alto. El paper LVIS usa
   t=0.001, pero en este dataset las clases raras aparecen en un % de imágenes
   más alto, así que el default se calibró a t=0.2 (articulado ~6.8x, omnibus
   ~2.4x, microbus ~2.1x; clases comunes quedan en 1x, inflado total ~1.2x).
3. Factor de repetición por imagen = el máximo de los r_c de las clases que
   contiene (una imagen "vale" por su clase más rara):
       r_i = max_{c in imagen} r_c    (1.0 si la imagen no tiene anotaciones)
4. Cada imagen se repite floor(r_i) veces, más 1 vez extra con probabilidad
   igual a la parte fraccionaria (redondeo estocástico, reproducible con semilla).

El resultado es un train.txt con rutas de imagen repetidas. Ultralytics acepta
`train:` apuntando a un .txt con rutas; las entradas duplicadas equivalen a
oversampling sin copiar archivos en disco.

Uso
---
    python -m src.oversample \
        --csv train.csv \
        --train-ids configs/train_ids.txt \
        --images-dir data/yolo/images/train \
        --out data/yolo/train_oversampled.txt \
        --thresh 0.2 \
        --max-repeat 50 \
        --seed 42

Luego, en configs/data.yaml apunta el split de train al .txt generado:
    train: train_oversampled.txt
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NUM_CLASSES = 9
CLASS_NAMES = [
    "auto", "combi", "microbus", "minibus", "omnibus",
    "articulado", "camion", "mototaxi", "motocicleta",
]


def parse_classes(target: str) -> set[int]:
    """Devuelve el conjunto de category_id (1-9) presentes en un frame.

    Conjunto vacío si el frame es "none" o no tiene detecciones válidas.
    """
    target = str(target).strip()
    if target.lower() == "none":
        return set()
    classes: set[int] = set()
    for token in target.split(";"):
        parts = token.strip().split()
        if len(parts) == 6:
            classes.add(int(parts[0]))
    return classes


def compute_class_image_freq(
    frame_classes: dict[str, set[int]],
) -> dict[int, float]:
    """Frecuencia a nivel de imagen f_c para cada clase (1-9).

    f_c = nº de imágenes que contienen la clase / nº total de imágenes.
    """
    n_images = len(frame_classes)
    img_count: Counter = Counter()
    for classes in frame_classes.values():
        for c in classes:
            img_count[c] += 1
    return {c: img_count.get(c, 0) / n_images for c in range(1, NUM_CLASSES + 1)}


def class_repeat_factors(
    class_freq: dict[int, float],
    thresh: float,
) -> dict[int, float]:
    """Factor de repetición por clase r_c = max(1, sqrt(t / f_c))."""
    r_c: dict[int, float] = {}
    for c in range(1, NUM_CLASSES + 1):
        f = class_freq.get(c, 0.0)
        if f <= 0.0:
            r_c[c] = 1.0  # clase ausente en el split: no influye
        else:
            r_c[c] = max(1.0, math.sqrt(thresh / f))
    return r_c


def image_repeat_factor(
    classes: set[int],
    r_c: dict[int, float],
) -> float:
    """Factor de repetición por imagen = máximo r_c de sus clases (1.0 si vacía)."""
    if not classes:
        return 1.0
    return max(r_c[c] for c in classes)


def stochastic_repeats(factor: float, rng: random.Random, max_repeat: int) -> int:
    """Convierte un factor flotante en un nº entero de repeticiones.

    floor(factor) repeticiones seguras + 1 extra con probabilidad = parte
    fraccionaria. El resultado se acota a [1, max_repeat].
    """
    factor = min(factor, float(max_repeat))
    base = math.floor(factor)
    frac = factor - base
    reps = base + (1 if rng.random() < frac else 0)
    return max(1, reps)


def build_oversampled_list(
    csv_path: Path,
    train_ids: list[str],
    images_dir: Path,
    thresh: float = 0.001,
    max_repeat: int = 50,
    seed: int = 42,
    img_ext: str = ".jpg",
) -> tuple[list[str], dict[int, float], dict[int, float]]:
    """Construye la lista de rutas de imagen con oversampling RFS.

    Returns:
        (lineas, class_freq, r_c)
        lineas    → rutas absolutas de imagen, con repeticiones aplicadas
        class_freq→ frecuencia f_c por clase (para el reporte)
        r_c       → factor de repetición por clase (para el reporte)
    """
    rng = random.Random(seed)

    df = pd.read_csv(csv_path)
    id_col = next(c for c in df.columns if c.lower() == "id")
    tgt_col = next(c for c in df.columns if c.lower() == "target")
    target_by_id: dict[str, str] = dict(zip(df[id_col].astype(str), df[tgt_col].astype(str)))

    train_set = set(train_ids)
    frame_classes: dict[str, set[int]] = {
        fid: parse_classes(target_by_id.get(fid, "none"))
        for fid in train_set
    }

    class_freq = compute_class_image_freq(frame_classes)
    r_c = class_repeat_factors(class_freq, thresh)

    lines: list[str] = []
    reps_per_class: Counter = Counter()  # nº de imágenes (con repes) que tocan cada clase
    for fid in train_ids:
        classes = frame_classes.get(fid, set())
        r_i = image_repeat_factor(classes, r_c)
        reps = stochastic_repeats(r_i, rng, max_repeat)
        img_path = str((images_dir / f"{fid}{img_ext}").resolve()).replace("\\", "/")
        lines.extend([img_path] * reps)
        for c in classes:
            reps_per_class[c] += reps

    return lines, class_freq, r_c


def report(
    n_original: int,
    lines: list[str],
    class_freq: dict[int, float],
    r_c: dict[int, float],
) -> None:
    print("\n" + "=" * 70)
    print("REPORTE DE OVERSAMPLING (Repeat Factor Sampling)")
    print("=" * 70)
    print(f"  Frames originales (train) : {n_original:>8,}")
    print(f"  Frames tras oversampling  : {len(lines):>8,}")
    print(f"  Factor de inflado total   : {len(lines) / max(1, n_original):>8.2f}x")
    print()
    print(f"  {'Clase':<13} {'f_c (img%)':>11} {'r_c (repite)':>13}")
    print("  " + "-" * 39)
    for c in range(1, NUM_CLASSES + 1):
        print(f"  {CLASS_NAMES[c-1]:<13} {100*class_freq.get(c,0):>10.3f}% {r_c.get(c,1):>12.2f}x")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oversampling de clases raras (Repeat Factor Sampling) para YOLO-OBB"
    )
    parser.add_argument("--csv", default=Path("train.csv"), type=Path,
                        help="train.csv con las anotaciones")
    parser.add_argument("--train-ids", default=Path("configs/train_ids.txt"), type=Path,
                        help="Archivo con los frame_ids del split de train (uno por línea)")
    parser.add_argument("--images-dir", default=Path("data/yolo/images/train"), type=Path,
                        help="Directorio donde están las imágenes de train")
    parser.add_argument("--out", default=Path("data/yolo/train_oversampled.txt"), type=Path,
                        help="Archivo .txt de salida con rutas (repetidas)")
    parser.add_argument("--thresh", default=0.2, type=float,
                        help="Umbral t de RFS; mayor t → más oversampling de raras")
    parser.add_argument("--max-repeat", default=50, type=int,
                        help="Tope de repeticiones por imagen (evita inflado excesivo)")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    train_ids = [
        line.strip()
        for line in args.train_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log.info("Train IDs leídos: %d", len(train_ids))

    lines, class_freq, r_c = build_oversampled_list(
        csv_path=args.csv,
        train_ids=train_ids,
        images_dir=args.images_dir,
        thresh=args.thresh,
        max_repeat=args.max_repeat,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Lista con oversampling escrita en %s (%d líneas)", args.out, len(lines))

    report(len(train_ids), lines, class_freq, r_c)


if __name__ == "__main__":
    main()
