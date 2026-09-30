"""Train and test a quarter-channel YOLOv8n-seg experiment without overwriting other runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/apple_quality_160_yolo_seg/data.yaml"))
    parser.add_argument("--model", type=Path, default=Path("cloud_train/yolov8n_seg_quarter_channels.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("yolov8n-seg.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("local_results"))
    parser.add_argument("--name", default="yolov8n_quarter_channels_160")
    return parser.parse_args()


def make_data_yaml(source_yaml: Path, destination: Path) -> Path:
    dataset_root = source_yaml.resolve().parent
    destination.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "names:",
                "  0: apple",
                "  1: apple-defect",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    cfg = parse_args()
    for required in (cfg.data, cfg.model, cfg.weights):
        if not required.is_file():
            raise FileNotFoundError(f"Required file is missing: {required}")
    if cfg.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")

    project = cfg.project.resolve()
    run_dir = project / cfg.name
    test_dir = project / f"{cfg.name}_test"
    if run_dir.exists() or test_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {run_dir} or {test_dir}")

    config_dir = project / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = make_data_yaml(cfg.data, config_dir / f"{cfg.name}.data.yaml")

    model = YOLO(str(cfg.model))
    # Channel shapes differ from standard YOLOv8n, so only shape-compatible
    # tensors are transferred. Ultralytics prints the exact transfer count.
    model.load(str(cfg.weights))
    info = model.info(imgsz=160, verbose=True)
    flops = float(info[-1]) if isinstance(info, (tuple, list)) and len(info) >= 4 else None
    config_dir.joinpath(f"{cfg.name}.model_info.txt").write_text(
        f"Ultralytics model.info return value: {info}\n"
        f"Initialization weights: {cfg.weights.resolve()}\n"
        "The model uses partial transfer because its channel shapes differ from standard YOLOv8n-seg.\n",
        encoding="utf-8",
    )

    model.train(
        data=str(data_yaml),
        imgsz=160,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        project=str(project),
        name=cfg.name,
        exist_ok=False,
        pretrained=False,
        seed=42,
        deterministic=True,
        patience=30,
        plots=True,
    )

    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"Best checkpoint was not created: {best}")

    metrics = YOLO(str(best)).val(
        data=str(data_yaml),
        split="test",
        imgsz=160,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        project=str(project),
        name=f"{cfg.name}_test",
        exist_ok=False,
        plots=True,
    )
    report = {
        "model": "YOLOv8n-seg with approximately quarter YOLOv8n channels",
        "input_size": [160, 160],
        "epochs": cfg.epochs,
        "initialization": str(cfg.weights.resolve()),
        "best_weights": str(best),
        "test_box_precision": float(metrics.box.mp),
        "test_box_recall": float(metrics.box.mr),
        "test_box_map50": float(metrics.box.map50),
        "test_box_map50_95": float(metrics.box.map),
        "test_mask_precision": float(metrics.seg.mp),
        "test_mask_recall": float(metrics.seg.mr),
        "test_mask_map50": float(metrics.seg.map50),
        "test_mask_map50_95": float(metrics.seg.map),
        "ultralytics_flops_at_160": flops,
    }
    run_dir.joinpath("experiment_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
