"""Train and evaluate YOLOv8n-seg on the prepared apple defect dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/apple_quality_160_yolo_seg/data.yaml"),
        help="Path to the generated YOLO data.yaml file.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size; lower it if CUDA runs out of memory.")
    parser.add_argument("--device", default="0", help="CUDA GPU index, e.g. 0; use cpu only when needed.")
    parser.add_argument("--workers", type=int, default=4, help="Data-loader worker processes.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolov8n-seg.pt"),
        help="Local pretrained checkpoint. If absent, Ultralytics downloads this filename automatically.",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/segment"), help="Directory for results.")
    parser.add_argument("--name", default="apple_yolov8n_160", help="Experiment subdirectory name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Could not find {args.data}. Upload the apple_quality_160_yolo_seg directory first."
        )
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the server NVIDIA driver and CUDA-enabled PyTorch installation.")

    # The prepared data.yaml contains the Windows source path. Build an equivalent
    # server-local configuration next to this run so uploads work on Linux unchanged.
    dataset_root = args.data.resolve().parent
    run_directory = args.project / args.name
    run_directory.mkdir(parents=True, exist_ok=True)
    server_data_yaml = run_directory / "data.server.yaml"
    server_data_yaml.write_text(
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

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(str(args.weights))
    model.train(
        data=str(server_data_yaml),
        imgsz=160,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        seed=42,
        deterministic=True,
        patience=30,
        plots=True,
    )

    best_weights = args.project / args.name / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training ended but best weights are missing: {best_weights}")

    # Test data has not been used for model selection. Evaluate it only once here.
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=str(server_data_yaml),
        split="test",
        imgsz=160,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=f"{args.name}_test",
        exist_ok=True,
        plots=True,
    )
    print("Test box mAP50-95:", metrics.box.map)
    print("Test mask mAP50-95:", metrics.seg.map)
    print(f"Best weights: {best_weights}")


if __name__ == "__main__":
    main()
