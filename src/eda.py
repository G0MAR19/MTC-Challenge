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
EDA profundo del dataset SMART Challenge (Fase 1).

Análisis generados:
  1.  Distribución de clases (instancias, % y nº de videos por clase).
  2.  Frecuencia por video (frames, detecciones, clases presentes).
  3.  Estadísticas de tamaño (width, height, area) globales y por clase.
  4.  Distribución de ángulos OBB: cruda [0,360) y plegada [0,180).
  5.  Distribución de aspect ratio por clase.
  6.  Heatmaps de posiciones (cx, cy): global y por clase.
  7.  Vehículos por frame (incluye frames "none" = 0).
  8.  Frames por clip.
  9.  Correlación clase-tamaño (boxplots de área).
  10. Correlación clase-ángulo (% de cajas rotadas por clase).

Salidas en outputs/eda/:
  *.png  → gráficos
  *.csv  → tablas exportadas
  report.html → reporte navegable que integra todo

Uso:
    python -m src.eda --csv train.csv --out outputs/eda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: solo escribe archivos

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

IMG_W, IMG_H = 1920, 1080
NUM_CLASSES = 9
CLASS_NAMES = {
    1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
    6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta",
}
RARE_CLASSES = [2, 3, 5, 6, 8]  # combi, microbus, omnibus, articulado, mototaxi
ANGLE_TOL = 0.5  # grados: tolerancia para considerar una caja "axial" (0/90/180/270)

sns.set_theme(style="whitegrid", font_scale=0.9)


# ── Carga y parseo ─────────────────────────────────────────────────────────

def load_detections(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parsea train.csv a formato largo: una fila por detección.

    Returns:
        (dets, frames)
        dets   → DataFrame con columnas: frame_id, video_id, category_id,
                 class_name, cx, cy, w, h, angle_deg, area, aspect_ratio
        frames → DataFrame por frame: frame_id, video_id, n_dets
    """
    df = pd.read_csv(csv_path)
    id_col = next(c for c in df.columns if c.lower() == "id")
    tgt_col = next(c for c in df.columns if c.lower() == "target")

    rows: list[tuple] = []
    frame_rows: list[tuple] = []
    for fid, target in zip(df[id_col].astype(str), df[tgt_col].astype(str)):
        vid = fid.rsplit("_", 1)[0]
        target = target.strip()
        n_dets = 0
        if target.lower() != "none":
            for tok in target.split(";"):
                parts = tok.strip().split()
                if len(parts) != 6:
                    continue
                cat = int(parts[0])
                cx, cy, w, h, ang = (float(x) for x in parts[1:])
                rows.append((fid, vid, cat, cx, cy, w, h, ang))
                n_dets += 1
        frame_rows.append((fid, vid, n_dets))

    dets = pd.DataFrame(
        rows, columns=["frame_id", "video_id", "category_id", "cx", "cy", "w", "h", "angle_deg"]
    )
    dets["class_name"] = dets["category_id"].map(CLASS_NAMES)
    dets["area"] = dets["w"] * dets["h"]
    # Aspect ratio invariante a la orientación de la caja: lado mayor / lado menor
    dets["aspect_ratio"] = np.maximum(dets["w"], dets["h"]) / np.minimum(dets["w"], dets["h"]).clip(lower=1e-6)
    frames = pd.DataFrame(frame_rows, columns=["frame_id", "video_id", "n_dets"])

    log.info("Frames: %d | Videos: %d | Detecciones: %d",
             len(frames), frames["video_id"].nunique(), len(dets))
    return dets, frames


# ── Tablas ─────────────────────────────────────────────────────────────────

def class_stats_table(dets: pd.DataFrame) -> pd.DataFrame:
    """Tabla maestra por clase: frecuencia, tamaño y rotación."""
    rows = []
    total = len(dets)
    for cat in range(1, NUM_CLASSES + 1):
        sub = dets[dets["category_id"] == cat]
        folded = sub["angle_deg"] % 90.0
        rotated = ((folded > ANGLE_TOL) & (folded < 90.0 - ANGLE_TOL)).mean() if len(sub) else 0.0
        rows.append({
            "category_id": cat,
            "clase": CLASS_NAMES[cat],
            "instancias": len(sub),
            "pct_instancias": 100 * len(sub) / total,
            "videos": sub["video_id"].nunique(),
            "frames": sub["frame_id"].nunique(),
            "w_mediana": sub["w"].median(),
            "h_mediana": sub["h"].median(),
            "area_mediana": sub["area"].median(),
            "area_p10": sub["area"].quantile(0.10) if len(sub) else np.nan,
            "area_p90": sub["area"].quantile(0.90) if len(sub) else np.nan,
            "aspect_ratio_mediana": sub["aspect_ratio"].median(),
            "pct_rotadas": 100 * rotated,
        })
    return pd.DataFrame(rows)


def video_stats_table(dets: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Tabla por video: frames, detecciones y clases presentes."""
    g_frames = frames.groupby("video_id").agg(n_frames=("frame_id", "count"))
    g_dets = dets.groupby("video_id").agg(
        n_dets=("frame_id", "count"),
        n_clases=("category_id", "nunique"),
    )
    classes_per_video = (
        dets.groupby("video_id")["class_name"]
        .apply(lambda s: ",".join(sorted(s.unique())))
        .rename("clases")
    )
    out = g_frames.join([g_dets, classes_per_video]).fillna({"n_dets": 0, "n_clases": 0, "clases": ""})
    out["n_dets"] = out["n_dets"].astype(int)
    out["n_clases"] = out["n_clases"].astype(int)
    out["dets_por_frame"] = out["n_dets"] / out["n_frames"]
    return out.reset_index().sort_values("n_dets", ascending=False)


# ── Gráficos ───────────────────────────────────────────────────────────────

def plot_class_distribution(stats: pd.DataFrame, out: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    order = stats.sort_values("instancias", ascending=False)
    sns.barplot(data=order, x="clase", y="instancias", hue="clase", legend=False,
                palette="viridis", ax=axes[0])
    axes[0].set_title("Instancias por clase (escala lineal)")
    axes[0].tick_params(axis="x", rotation=45)
    sns.barplot(data=order, x="clase", y="instancias", hue="clase", legend=False,
                palette="viridis", ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_title("Instancias por clase (escala log — revela las raras)")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        for c in ax.containers:
            ax.bar_label(c, fmt="%.0f", fontsize=7)
    fig.tight_layout()
    name = "01_class_distribution.png"
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


def plot_videos_per_class(stats: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    order = stats.sort_values("videos", ascending=False)
    sns.barplot(data=order, x="clase", y="videos", hue="clase", legend=False,
                palette="mako", ax=ax)
    ax.set_title("Nº de videos en los que aparece cada clase (de 1088)")
    ax.tick_params(axis="x", rotation=45)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", fontsize=8)
    fig.tight_layout()
    name = "02_videos_per_class.png"
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


def plot_size_by_class(dets: pd.DataFrame, out: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in zip(
        axes, ["w", "h", "area"], ["Width (px)", "Height (px)", "Área (px²)"]
    ):
        sns.boxplot(data=dets, x="class_name", y=col, hue="class_name", legend=False,
                    palette="crest", showfliers=False, ax=ax)
        ax.set_title(f"{title} por clase")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    axes[2].set_yscale("log")
    fig.suptitle("Correlación clase-tamaño", y=1.02)
    fig.tight_layout()
    name = "03_size_by_class.png"
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_angle_distribution(dets: pd.DataFrame, out: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(dets["angle_deg"], bins=72, color="#3b6ea5", edgecolor="white")
    axes[0].set_title("Ángulos crudos [0°, 360°)")
    axes[0].set_xlabel("angle_deg")
    axes[1].hist(dets["angle_deg"] % 180.0, bins=36, color="#a53b3b", edgecolor="white")
    axes[1].set_title("Ángulos plegados a [0°, 180°) — equivalencia OBB")
    axes[1].set_xlabel("angle_deg % 180")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("instancias (log)")
    fig.tight_layout()
    name = "04_angle_distribution.png"
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


def plot_rotated_by_class(stats: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    order = stats.sort_values("pct_rotadas", ascending=False)
    sns.barplot(data=order, x="clase", y="pct_rotadas", hue="clase", legend=False,
                palette="rocket", ax=ax)
    ax.set_title("Correlación clase-ángulo: % de cajas NO axiales por clase")
    ax.set_ylabel("% rotadas")
    ax.tick_params(axis="x", rotation=45)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", fontsize=8)
    fig.tight_layout()
    name = "05_rotated_by_class.png"
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


def plot_aspect_ratio(dets: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=dets, x="class_name", y="aspect_ratio", hue="class_name",
                legend=False, palette="flare", showfliers=False, ax=ax)
    ax.set_title("Aspect ratio (lado mayor / lado menor) por clase")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    name = "06_aspect_ratio_by_class.png"
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


def plot_position_heatmaps(dets: pd.DataFrame, out: Path) -> list[str]:
    """Heatmap global de posiciones (cx, cy) + grid 3x3 por clase."""
    names: list[str] = []

    fig, ax = plt.subplots(figsize=(8, 4.8))
    hist, _, _ = np.histogram2d(
        dets["cy"], dets["cx"], bins=[54, 96], range=[[0, IMG_H], [0, IMG_W]]
    )
    ax.imshow(np.log1p(hist), cmap="inferno", extent=[0, IMG_W, IMG_H, 0], aspect="auto")
    ax.set_title("Heatmap global de posiciones (log) — 1920×1080")
    fig.tight_layout()
    name = "07_heatmap_global.png"
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    names.append(name)

    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    for ax, cat in zip(axes.flat, range(1, NUM_CLASSES + 1)):
        sub = dets[dets["category_id"] == cat]
        hist, _, _ = np.histogram2d(
            sub["cy"], sub["cx"], bins=[27, 48], range=[[0, IMG_H], [0, IMG_W]]
        )
        ax.imshow(np.log1p(hist), cmap="inferno", extent=[0, IMG_W, IMG_H, 0], aspect="auto")
        ax.set_title(f"{CLASS_NAMES[cat]} (n={len(sub):,})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Heatmap de posiciones por clase (log)")
    fig.tight_layout()
    name = "08_heatmap_por_clase.png"
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    names.append(name)
    return names


def plot_dets_per_frame(frames: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(frames["n_dets"], bins=range(0, frames["n_dets"].max() + 2),
            color="#2a7f62", edgecolor="white")
    ax.set_title(
        f"Vehículos por frame — media {frames['n_dets'].mean():.1f}, "
        f"frames 'none': {(frames['n_dets'] == 0).sum():,} "
        f"({100 * (frames['n_dets'] == 0).mean():.1f}%)"
    )
    ax.set_xlabel("detecciones por frame")
    ax.set_ylabel("frames")
    fig.tight_layout()
    name = "09_dets_per_frame.png"
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


def plot_frames_per_clip(frames: pd.DataFrame, out: Path) -> str:
    per_clip = frames.groupby("video_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_clip, bins=range(per_clip.min(), per_clip.max() + 2),
            color="#7f2a5e", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title(f"Frames por clip — {len(per_clip)} clips, moda {per_clip.mode()[0]}")
    ax.set_xlabel("frames por clip")
    ax.set_ylabel("clips (log)")
    fig.tight_layout()
    name = "10_frames_per_clip.png"
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


# ── Reporte HTML ───────────────────────────────────────────────────────────

def write_html_report(
    out: Path,
    stats: pd.DataFrame,
    frames: pd.DataFrame,
    images: list[str],
) -> None:
    """Genera report.html con tablas resumen e imágenes incrustadas."""
    n_frames = len(frames)
    n_videos = frames["video_id"].nunique()
    n_dets = int(stats["instancias"].sum())
    n_none = int((frames["n_dets"] == 0).sum())

    stats_fmt = stats.copy()
    for col in stats_fmt.select_dtypes("float").columns:
        stats_fmt[col] = stats_fmt[col].round(2)

    html = [
        "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>",
        "<title>EDA — SMART Challenge 2026</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:2em auto;color:#222}"
        "h1{border-bottom:3px solid #3b6ea5}h2{color:#3b6ea5;margin-top:1.6em}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:4px 8px}"
        "th{background:#3b6ea5;color:#fff}tr:nth-child(even){background:#f4f7fb}"
        "img{max-width:100%;border:1px solid #ddd;margin:8px 0}"
        ".kpi{display:inline-block;background:#f4f7fb;border:1px solid #cdd;border-radius:6px;"
        "padding:10px 18px;margin:4px;font-size:15px}</style></head><body>",
        "<h1>EDA profundo — SMART Challenge 2026</h1>",
        f"<div><span class='kpi'><b>{n_frames:,}</b> frames</span>"
        f"<span class='kpi'><b>{n_videos:,}</b> videos</span>"
        f"<span class='kpi'><b>{n_dets:,}</b> detecciones</span>"
        f"<span class='kpi'><b>{n_none:,}</b> frames none</span></div>",
        "<h2>Tabla maestra por clase</h2>",
        stats_fmt.to_html(index=False),
    ]
    for img in images:
        html.append(f"<h2>{img.split('.')[0].replace('_', ' ')}</h2><img src='{img}'>")
    html.append("</body></html>")
    (out / "report.html").write_text("\n".join(html), encoding="utf-8")
    log.info("Reporte HTML: %s", out / "report.html")


# ── Main ───────────────────────────────────────────────────────────────────

def run_eda(csv_path: Path, out_dir: Path) -> None:
    """Ejecuta el EDA completo y escribe todos los artefactos en out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dets, frames = load_detections(csv_path)

    stats = class_stats_table(dets)
    vstats = video_stats_table(dets, frames)
    stats.to_csv(out_dir / "class_stats.csv", index=False)
    vstats.to_csv(out_dir / "video_stats.csv", index=False)
    log.info("CSVs exportados: class_stats.csv, video_stats.csv")

    images: list[str] = []
    images.append(plot_class_distribution(stats, out_dir))
    images.append(plot_videos_per_class(stats, out_dir))
    images.append(plot_size_by_class(dets, out_dir))
    images.append(plot_angle_distribution(dets, out_dir))
    images.append(plot_rotated_by_class(stats, out_dir))
    images.append(plot_aspect_ratio(dets, out_dir))
    images.extend(plot_position_heatmaps(dets, out_dir))
    images.append(plot_dets_per_frame(frames, out_dir))
    images.append(plot_frames_per_clip(frames, out_dir))
    log.info("Gráficos generados: %d", len(images))

    write_html_report(out_dir, stats, frames, images)

    # Hallazgos clave por consola
    print("\n" + "=" * 64)
    print("HALLAZGOS CLAVE")
    print("=" * 64)
    rare = stats[stats["category_id"].isin(RARE_CLASSES)]
    for _, r in rare.iterrows():
        print(f"  {r['clase']:<12} {r['instancias']:>7,} inst | {r['videos']:>4} videos "
              f"| área mediana {r['area_mediana']:>8,.0f} px² | {r['pct_rotadas']:.1f}% rotadas")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA profundo del dataset MTC")
    parser.add_argument("--csv", default=Path("train.csv"), type=Path)
    parser.add_argument("--out", default=Path("outputs/eda"), type=Path)
    args = parser.parse_args()
    run_eda(args.csv, args.out)
