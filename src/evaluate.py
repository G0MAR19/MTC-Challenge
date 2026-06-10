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
Calcula la métrica oficial: Macro AP-rIoU @ [0.50 : 0.80] (step 0.05).

Implementación local para validar antes de subir a Kaggle.

Uso:
    python -m src.evaluate --gt data/raw/train.csv \
                           --preds outputs/val_predictions.json \
                           --ids outputs/val_ids.txt
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import math

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

IOU_THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 0.85, 0.05)]
NUM_CLASSES = 9
CATEGORY_OFFSET = 1


def obb_to_polygon(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Polygon:
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    pts = [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]
    return Polygon(pts)


def riou(box_a: list[float], box_b: list[float]) -> float:
    """rIoU entre dos OBBs [cx, cy, w, h, angle_deg]."""
    try:
        pa = obb_to_polygon(*box_a)
        pb = obb_to_polygon(*box_b)
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """Área bajo la curva P-R usando interpolación de 101 puntos (COCO style)."""
    prec = np.array(precisions)
    rec = np.array(recalls)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = rec >= t
        ap += prec[mask].max() if mask.any() else 0.0
    return ap / 101


def compute_class_ap(
    gt_by_frame: dict[str, list[list[float]]],
    pred_by_frame: dict[str, list[tuple[float, list[float]]]],
    iou_thresh: float,
) -> float:
    """AP para una clase a un umbral rIoU dado."""
    total_gt = sum(len(gts) for gts in gt_by_frame.values())
    if total_gt == 0:
        return float("nan")

    # Recoge TODAS las predicciones de esta clase (incluye frames sin GT = FP)
    all_preds: list[tuple[float, str, int]] = []
    for frame_id, frame_preds in pred_by_frame.items():
        for local_idx, (score, _box) in enumerate(frame_preds):
            all_preds.append((score, frame_id, local_idx))

    if not all_preds:
        return 0.0

    all_preds.sort(key=lambda x: -x[0])
    matched_gt: dict[str, set[int]] = defaultdict(set)

    tps, fps = [], []
    for score, frame_id, local_idx in all_preds:
        pred_box = pred_by_frame[frame_id][local_idx][1]
        gts = gt_by_frame.get(frame_id, [])
        best_iou, best_j = -1.0, -1
        for j, gt_box in enumerate(gts):
            if j in matched_gt[frame_id]:
                continue
            iou_val = riou(pred_box, gt_box)
            if iou_val > best_iou:
                best_iou, best_j = iou_val, j
        if best_iou >= iou_thresh and best_j >= 0:
            matched_gt[frame_id].add(best_j)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls = cum_tp / total_gt
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    return compute_ap(list(precisions), list(recalls))


def macro_ap_riou(
    gt_df: pd.DataFrame,
    predictions: dict[str, list[dict]],
    frame_ids: list[str],
) -> float:
    """Calcula Macro AP-rIoU @ [0.50:0.80] sobre los frame_ids dados."""
    # Organizar GT por clase y frame
    gt_by_class_frame: dict[int, dict[str, list[list[float]]]] = {
        c: defaultdict(list) for c in range(1, NUM_CLASSES + 1)
    }
    for _, row in gt_df[gt_df["Id"].isin(frame_ids)].iterrows():
        frame_id = row["Id"]
        target = str(row["Target"])
        if target.strip().lower() == "none":
            continue
        for token in target.strip().split(";"):
            parts = token.strip().split()
            if len(parts) != 6:
                continue
            cat_id = int(parts[0])
            box = [float(x) for x in parts[1:]]
            gt_by_class_frame[cat_id][frame_id].append(box)

    # Organizar predicciones por clase y frame
    pred_by_class_frame: dict[int, dict[str, list[tuple[float, list[float]]]]] = {
        c: defaultdict(list) for c in range(1, NUM_CLASSES + 1)
    }
    for frame_id in frame_ids:
        for det in predictions.get(frame_id, []):
            cat_id = det["category_id"]
            box = [det["cx"], det["cy"], det["w"], det["h"], det["angle_deg"]]
            pred_by_class_frame[cat_id][frame_id].append((det["score"], box))

    aps_per_class = []
    for cat_id in range(1, NUM_CLASSES + 1):
        aps_per_thresh = []
        for iou_t in IOU_THRESHOLDS:
            ap = compute_class_ap(
                gt_by_class_frame[cat_id],
                pred_by_class_frame[cat_id],
                iou_t,
            )
            if not np.isnan(ap):
                aps_per_thresh.append(ap)
        class_ap = float(np.mean(aps_per_thresh)) if aps_per_thresh else 0.0
        log.info("Clase %d AP: %.4f", cat_id, class_ap)
        aps_per_class.append(class_ap)

    macro_ap = float(np.mean(aps_per_class))
    log.info("Macro AP-rIoU: %.4f", macro_ap)
    return macro_ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa Macro AP-rIoU localmente")
    parser.add_argument("--gt", required=True, type=Path, help="train.csv con Ground Truth")
    parser.add_argument("--preds", required=True, type=Path, help="JSON con predicciones")
    parser.add_argument("--ids", required=True, type=Path, help="Archivo .txt con frame_ids de val")
    args = parser.parse_args()

    gt_df = pd.read_csv(args.gt)
    with open(args.preds) as f:
        preds = json.load(f)
    frame_ids = Path(args.ids).read_text().strip().splitlines()

    score = macro_ap_riou(gt_df, preds, frame_ids)
    print(f"Macro AP-rIoU @ [0.50:0.80] = {score:.4f}")
