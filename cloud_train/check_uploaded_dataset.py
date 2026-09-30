"""Validate a cloud upload of the prepared apple YOLOv8 segmentation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


EXPECTED_IMAGES = {"train": 1938, "val": 185, "test": 93}
EXPECTED_SIZE = (160, 160)
VALID_CLASSES = {"0", "1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/apple_quality_160_yolo_seg"),
        help="Uploaded dataset directory containing train/, val/, and test/.",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Optional yolov8n-seg.pt path to check.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    errors: list[str] = []
    class_counts: Counter[str] = Counter()

    if not root.is_dir():
        print(f"ERROR: Dataset directory does not exist: {root}")
        return 1
    if not (root / "data.yaml").is_file():
        errors.append("Missing data.yaml")

    for split, expected_count in EXPECTED_IMAGES.items():
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: missing images/ or labels/ directory")
            continue

        images = sorted(image_dir.glob("*.jpg"))
        print(f"{split}: {len(images)} image(s), expected {expected_count}")
        if len(images) != expected_count:
            errors.append(f"{split}: image count is {len(images)}, expected {expected_count}")

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"Missing label: {label_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    if image.size != EXPECTED_SIZE:
                        errors.append(f"Wrong dimensions {image.size}: {image_path}")
            except Exception as exc:
                errors.append(f"Unreadable image {image_path}: {exc}")
                continue

            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split()
                if len(fields) < 7 or len(fields) % 2 == 0 or fields[0] not in VALID_CLASSES:
                    errors.append(f"Invalid segmentation label {label_path}:{line_number}")
                    continue
                try:
                    coordinates = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(f"Non-numeric label {label_path}:{line_number}")
                    continue
                if any(value < 0 or value > 1 for value in coordinates):
                    errors.append(f"Out-of-range coordinate {label_path}:{line_number}")
                class_counts[fields[0]] += 1

    if args.weights is not None:
        if not args.weights.is_file() or args.weights.stat().st_size < 1_000_000:
            errors.append(f"Missing or suspiciously small model weights: {args.weights}")
        else:
            print(f"weights: present ({args.weights.stat().st_size:,} bytes)")

    print("annotation instances:", json.dumps(dict(sorted(class_counts.items()))))
    if errors:
        print(f"\nFAILED: found {len(errors)} problem(s).")
        for error in errors[:30]:
            print(f"- {error}")
        if len(errors) > 30:
            print(f"- ... plus {len(errors) - 30} more")
        return 1

    print("\nPASSED: upload is complete and the dataset is ready for YOLOv8 segmentation training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
