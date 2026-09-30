"""Collect test-set Box metrics for apple and apple-defect without using the GPU."""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "apple_quality_160_yolo_seg" / "data.yaml"
MODELS = {
    "original": ROOT / "local_results" / "yolov8n_original_160_rtx2050" / "weights" / "best.pt",
    "mobilenetv2": ROOT / "local_results" / "mobilenetv2_160_rtx2050" / "weights" / "best.pt",
    "middle_channels": ROOT / "local_results" / "yolov8n_three_quarter_channels_160_rtx2050" / "weights" / "best.pt",
    "half_channels": ROOT / "local_results" / "yolov8n_half_channels_160_rtx2050" / "weights" / "best.pt",
}


def main() -> None:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for name, path in MODELS.items():
        print(f"Evaluating {name} on CPU: {path}", flush=True)
        model = YOLO(str(path))
        metrics = model.val(
            data=str(DATA), split="test", imgsz=160, batch=8, device="cpu", workers=0,
            plots=False, project=str(ROOT / "local_results" / "_metrics_temp"), name=name,
            exist_ok=True,
        )
        output[name] = {
            metrics.names[i]: {
                "precision": float(metrics.box.class_result(i)[0]),
                "recall": float(metrics.box.class_result(i)[1]),
                "map50": float(metrics.box.class_result(i)[2]),
                "map50_95": float(metrics.box.class_result(i)[3]),
            }
            for i in metrics.names
        }
    path = ROOT / "local_results" / "box_per_class_metrics_160.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
