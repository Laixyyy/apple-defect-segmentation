"""Evaluate a saved YOLO segmentation checkpoint on the apple test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    output_dir = args.project.resolve() / args.name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing test results: {output_dir}")

    model = YOLO(str(args.model))
    info = model.info(imgsz=160, verbose=True)
    flops = float(info[-1]) if isinstance(info, (tuple, list)) and len(info) >= 4 else None
    metrics = model.val(
        data=str(args.data), split="test", imgsz=160, batch=args.batch,
        device=args.device, workers=args.workers, project=str(args.project),
        name=args.name, exist_ok=False, plots=True,
    )
    result = {
        "best_weights": str(args.model.resolve()),
        "input_size": [160, 160],
        "ultralytics_flops_at_160": flops,
        "test_box_precision": float(metrics.box.mp),
        "test_box_recall": float(metrics.box.mr),
        "test_box_map50": float(metrics.box.map50),
        "test_box_map50_95": float(metrics.box.map),
        "test_mask_precision": float(metrics.seg.mp),
        "test_mask_recall": float(metrics.seg.mr),
        "test_mask_map50": float(metrics.seg.map50),
        "test_mask_map50_95": float(metrics.seg.map),
        "test_mask_per_class": {
            metrics.names[i]: {
                "precision": float(metrics.seg.class_result(i)[0]),
                "recall": float(metrics.seg.class_result(i)[1]),
                "map50": float(metrics.seg.class_result(i)[2]),
                "map50_95": float(metrics.seg.class_result(i)[3]),
            } for i in metrics.names
        },
    }
    args.model.parent.parent.joinpath("experiment_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
