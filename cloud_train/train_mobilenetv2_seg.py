"""Train a separate MobileNetV2-backbone YOLO instance-segmentation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/apple_quality_160_yolo_seg/data.yaml'))
    parser.add_argument('--model', type=Path, default=Path('cloud_train/mobilenetv2_yolov8_seg.yaml'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=-1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', type=Path, default=Path('results'))
    parser.add_argument('--name', default='mobilenetv2_160')
    return parser.parse_args()


def make_server_yaml(source_yaml: Path, destination: Path) -> Path:
    dataset_root = source_yaml.resolve().parent
    destination.write_text(
        '\n'.join([
            f'path: {dataset_root.as_posix()}', 'train: train/images', 'val: val/images', 'test: test/images',
            'names:', '  0: apple', '  1: apple-defect', ''
        ]), encoding='utf-8')
    return destination


def main() -> None:
    cfg = args()
    if not cfg.data.is_file() or not cfg.model.is_file():
        raise FileNotFoundError('Upload both the dataset directory and cloud_train MobileNetV2 files.')
    if cfg.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this Python environment.')

    project = cfg.project.resolve()
    run_dir = project / cfg.name
    if run_dir.exists():
        raise FileExistsError(f'Refusing to overwrite an existing experiment: {run_dir}')
    config_dir = project / '_configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = make_server_yaml(cfg.data, config_dir / f'{cfg.name}.data.yaml')

    model = YOLO(str(cfg.model))
    # Prints the model size and GFLOPs before any training begins.
    info = model.info(imgsz=160, verbose=True)
    flops = float(info[-1]) if isinstance(info, (tuple, list)) and len(info) >= 4 else None
    config_dir.joinpath(f'{cfg.name}.model_info.txt').write_text(
        f'Ultralytics model.info return value: {info}\n', encoding='utf-8'
    )

    model.train(
        data=str(data_yaml), imgsz=160, epochs=cfg.epochs, batch=cfg.batch, device=cfg.device,
        workers=cfg.workers, project=str(project), name=cfg.name, exist_ok=False,
        pretrained=False, seed=42, deterministic=True, patience=30, plots=True,
    )

    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f'Best checkpoint was not created: {best}')
    eval_batch = 16 if cfg.batch == -1 else cfg.batch
    metrics = YOLO(str(best)).val(
        data=str(data_yaml), split='test', imgsz=160, batch=eval_batch, device=cfg.device,
        workers=cfg.workers, project=str(project), name=f'{cfg.name}_test', exist_ok=False, plots=True,
    )
    report = {
        'model': 'MobileNetV2 backbone + YOLO instance-segmentation head',
        'input_size': [160, 160], 'epochs': cfg.epochs, 'best_weights': str(best),
        'test_box_map50_95': float(metrics.box.map), 'test_mask_map50_95': float(metrics.seg.map),
        'test_box_map50': float(metrics.box.map50), 'test_mask_map50': float(metrics.seg.map50),
        'ultralytics_flops': flops,
        'flops_note': 'GFLOPs is produced by Ultralytics model.info at imgsz=160.',
    }
    best.parent.parent.joinpath('experiment_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
