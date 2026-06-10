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
Ejecuta inferencia YOLO-OBB sobre los frames del set de test y genera predicciones
en el formato esperado por submission.py.

Uso:
    python -m src.inference --weights runs/mtc_obb/weights/best.pt \
                            --test-dir data/raw/test \
                            --ids sample_submission.csv \
                            --out outputs/raw_predictions.csv \
                            --conf 0.15 --iou 0.45
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CATEGORY_OFFSET = 1  # YOLO 0-indexed → competition 1-indexed


def run_inference(
    weights: Path,
    test_dir: Path,
    frame_ids: list[str],
    conf: float = 0.15,
    iou: float = 0.45,
    max_det: int = 300,
    device: str | int = 0,
) -> dict[str, list[dict]]:
    """Corre el modelo sobre cada frame y devuelve predicciones por frame_id.

    Returns:
        dict mapping frame_id -> list of {score, category_id, cx, cy, w, h, angle}
    """
    model = YOLO(str(weights))
    predictions: dict[str, list[dict]] = {}

    for frame_id in tqdm(frame_ids, desc="Inferencia"):
        img_path = test_dir / f"{frame_id}.jpg"
        if not img_path.exists():
            img_path = test_dir / f"{frame_id}.png"

        if not img_path.exists():
            log.warning("Frame no encontrado: %s", frame_id)
            predictions[frame_id] = []
            continue

        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )

        dets = []
        for result in results:
            if result.obb is None:
                continue
            boxes = result.obb
            for i in range(len(boxes)):
                # xywhr: center_x, center_y, width, height, rotation (radians)
                xywhr = boxes.xywhr[i].cpu().numpy()
                conf_val = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu()) + CATEGORY_OFFSET
                cx, cy, w, h, angle_rad = xywhr
                angle_deg = math.degrees(float(angle_rad))
                dets.append({
                    "score": conf_val,
                    "category_id": cls_id,
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "angle_deg": angle_deg,
                })
        # Sort by score descending for top-N capping downstream
        dets.sort(key=lambda d: d["score"], reverse=True)
        predictions[frame_id] = dets

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia YOLO-OBB sobre test set")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--ids", default=Path("sample_submission.csv"), type=Path,
                        help="CSV con los Ids del test set (columna 'Id')")
    parser.add_argument("--out", default=Path("outputs/raw_predictions.csv"), type=Path)
    parser.add_argument("--conf", default=0.15, type=float)
    parser.add_argument("--iou", default=0.45, type=float)
    parser.add_argument("--max-det", default=300, type=int)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    ids_df = pd.read_csv(args.ids)
    frame_ids = ids_df["Id"].tolist()

    device = int(args.device) if args.device.isdigit() else args.device
    if isinstance(device, int) and not torch.cuda.is_available():
        log.warning("GPU no disponible, usando CPU")
        device = "cpu"

    preds = run_inference(
        args.weights, args.test_dir, frame_ids,
        conf=args.conf, iou=args.iou, max_det=args.max_det, device=device,
    )

    # Serializar para que submission.py pueda leerlo
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(preds, f)
    log.info("Predicciones guardadas en %s", args.out)
