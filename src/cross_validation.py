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
Genera 5 folds de validación cruzada agrupados por video_id.

Usa StratifiedGroupKFold de sklearn:
  - groups = video_id  → un video NUNCA aparece en train y val del mismo fold.
  - y      = clase más rara presente en cada frame → las clases escasas
    (articulado: 5 videos, omnibus, microbus) se reparten lo mejor posible
    entre los folds, que es lo que importa para la métrica Macro AP-rIoU.

Frames sin anotación ("none") se estratifican como clase 0 (fondo) para que
también queden repartidos de forma uniforme.

Salida (en --out-dir, por defecto configs/cv/):
  fold_0_train.csv ... fold_4_train.csv   ← mismas columnas que train.csv
  fold_0_val.csv   ... fold_4_val.csv
  cv_summary.csv                          ← instancias por clase en cada fold

Uso:
    python -m src.cross_validation --csv train.csv --n-folds 5 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = [
    "auto", "combi", "microbus", "minibus", "omnibus",
    "articulado", "camion", "mototaxi", "motocicleta",
]
NUM_CLASSES = 9
BACKGROUND = 0  # etiqueta de estratificación para frames "none"


# ── Extracción de etiquetas por frame ──────────────────────────────────────

def parse_frame_classes(target: str) -> list[int]:
    """Devuelve los category_id (1-9) presentes en el Target de un frame.

    Un frame "none" devuelve lista vacía.
    """
    t = str(target).strip()
    if t.lower() == "none":
        return []
    classes: list[int] = []
    for tok in t.split(";"):
        parts = tok.strip().split()
        if len(parts) == 6:
            classes.append(int(parts[0]))
    return classes


def build_strat_labels(df: pd.DataFrame) -> tuple[np.ndarray, Counter]:
    """Construye la etiqueta de estratificación de cada frame.

    La etiqueta es la clase MÁS RARA (por frecuencia global) presente en el
    frame; frames vacíos reciben BACKGROUND. Devuelve (labels, class_totals).
    """
    frame_classes: list[list[int]] = [
        parse_frame_classes(t) for t in df["Target"]
    ]

    class_totals: Counter = Counter()
    for classes in frame_classes:
        class_totals.update(classes)

    labels = np.array(
        [
            min(classes, key=lambda c: class_totals[c]) if classes else BACKGROUND
            for classes in frame_classes
        ],
        dtype=np.int64,
    )
    return labels, class_totals


# ── Generación de folds ────────────────────────────────────────────────────

def make_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Devuelve [(train_idx, val_idx), ...] con StratifiedGroupKFold.

    groups = video_id garantiza que ningún video se reparta entre train y val.
    """
    labels, class_totals = build_strat_labels(df)
    groups = df["video_id"].to_numpy()

    log.info(
        "Estratificando %d frames de %d videos en %d folds (seed=%d)",
        len(df), df["video_id"].nunique(), n_folds, seed,
    )
    for cls in sorted(class_totals, key=lambda c: class_totals[c]):
        n_videos = df.loc[labels == cls, "video_id"].nunique()
        log.info(
            "  clase %d (%s): %d instancias, etiqueta de %d frames / %d videos",
            cls, CLASS_NAMES[cls - 1], class_totals[cls],
            int((labels == cls).sum()), n_videos,
        )

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(df, labels, groups))


def verify_no_leakage(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Verifica que ningún video aparezca en train y val del mismo fold.

    Lanza AssertionError si hay fuga; el script no debe continuar en ese caso.
    """
    for k, (train_idx, val_idx) in enumerate(folds):
        train_videos = set(df.iloc[train_idx]["video_id"])
        val_videos = set(df.iloc[val_idx]["video_id"])
        overlap = train_videos & val_videos
        assert not overlap, (
            f"FUGA en fold {k}: {len(overlap)} videos en train y val: "
            f"{sorted(overlap)[:5]}..."
        )
        log.info(
            "Fold %d sin fuga: %d videos train, %d videos val, overlap=0",
            k, len(train_videos), len(val_videos),
        )


# ── Reporte y guardado ─────────────────────────────────────────────────────

def fold_class_counts(df: pd.DataFrame, idx: np.ndarray) -> Counter:
    """Cuenta instancias por clase (1-9) en los frames indicados."""
    counts: Counter = Counter()
    for target in df.iloc[idx]["Target"]:
        counts.update(parse_frame_classes(target))
    return counts


def save_folds(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> pd.DataFrame:
    """Escribe fold_{k}_{train,val}.csv y devuelve el DataFrame resumen."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_cols = ["Id", "Target"]
    summary_rows: list[dict[str, object]] = []

    for k, (train_idx, val_idx) in enumerate(folds):
        for split, idx in [("train", train_idx), ("val", val_idx)]:
            path = out_dir / f"fold_{k}_{split}.csv"
            df.iloc[idx][csv_cols].to_csv(path, index=False)
            log.info("Escrito %s (%d frames)", path, len(idx))

            counts = fold_class_counts(df, idx)
            summary_rows.append(
                {
                    "fold": k,
                    "split": split,
                    "frames": len(idx),
                    "videos": df.iloc[idx]["video_id"].nunique(),
                    **{
                        CLASS_NAMES[c - 1]: counts.get(c, 0)
                        for c in range(1, NUM_CLASSES + 1)
                    },
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "cv_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Resumen escrito en %s", summary_path)
    return summary


def report(summary: pd.DataFrame) -> None:
    """Imprime instancias de cada clase en el val de cada fold."""
    val = summary[summary["split"] == "val"].set_index("fold")

    print("\n" + "=" * 78)
    print("INSTANCIAS POR CLASE EN VAL DE CADA FOLD")
    print("=" * 78)
    header = f"  {'Clase':<13}" + "".join(f"  fold {k:<4}" for k in val.index)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in CLASS_NAMES:
        row = f"  {name:<13}" + "".join(f"{val.loc[k, name]:>9,}" for k in val.index)
        flag = "  <<< revisar" if (val[name] == 0).any() else ""
        print(row + flag)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'frames':<13}" + "".join(f"{val.loc[k, 'frames']:>9,}" for k in val.index))
    print(f"  {'videos':<13}" + "".join(f"{val.loc[k, 'videos']:>9,}" for k in val.index))
    print("=" * 78)

    zero_cells = [
        (name, k)
        for name in CLASS_NAMES
        for k in val.index
        if val.loc[k, name] == 0
    ]
    if zero_cells:
        print("\nAVISO: clases sin instancias en algún fold (inevitable si la")
        print("clase tiene menos videos que folds, p.ej. articulado con 5):")
        for name, k in zero_cells:
            print(f"  - {name} en fold {k}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-Fold estratificado y agrupado por video_id"
    )
    parser.add_argument("--csv",     default=Path("train.csv"),   type=Path)
    parser.add_argument("--out-dir", default=Path("configs/cv"),  type=Path)
    parser.add_argument("--n-folds", default=5,  type=int)
    parser.add_argument("--seed",    default=42, type=int)
    args = parser.parse_args()

    log.info("Cargando %s ...", args.csv)
    df = pd.read_csv(args.csv)
    id_col  = next(c for c in df.columns if c.lower() == "id")
    tgt_col = next(c for c in df.columns if c.lower() == "target")
    df = df.rename(columns={id_col: "Id", tgt_col: "Target"})
    df["video_id"] = df["Id"].str.rsplit("_", n=1).str[0]

    folds = make_folds(df, n_folds=args.n_folds, seed=args.seed)
    verify_no_leakage(df, folds)
    summary = save_folds(df, folds, args.out_dir)
    report(summary)


if __name__ == "__main__":
    main()
