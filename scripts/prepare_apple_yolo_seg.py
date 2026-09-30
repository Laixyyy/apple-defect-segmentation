"""Build a 160x160 YOLOv8 instance-segmentation dataset from the NDJSON export."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(r"D:\BIT\fruits")
SOURCE = ROOT / "data" / "raw_apple_quality"
MANIFEST = SOURCE / "apple-qualityv4icoco-segmentation.ndjson"
OUTPUT = ROOT / "data" / "apple_quality_160_yolo_seg"
SIZE = (160, 160)

# The export declares an unused duplicate defect class (ID 0). Only IDs 1 and 2
# occur in annotations, so remap them to a compact two-class YOLO dataset.
CLASS_MAP = {1: 0, 2: 1}
CLASS_NAMES = ["apple", "apple-defect"]


def yolo_line(segment: list[float]) -> str:
    source_class, *coordinates = segment
    if source_class not in CLASS_MAP:
        raise ValueError(f"Unexpected source class: {source_class}")
    if len(coordinates) < 6 or len(coordinates) % 2:
        raise ValueError("A polygon must contain at least three x/y pairs.")
    if any(not 0 <= float(value) <= 1 for value in coordinates):
        raise ValueError("Polygon coordinates must be normalized to [0, 1].")
    formatted = " ".join(f"{float(value):.6f}" for value in coordinates)
    return f"{CLASS_MAP[source_class]} {formatted}"


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest missing: {MANIFEST}")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    records = []
    with MANIFEST.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("type") == "image":
                records.append(item)

    image_counts: Counter[str] = Counter()
    segment_counts: Counter[str] = Counter()
    for item in records:
        split = item["split"]
        image_name = item["file"]
        source_image = SOURCE / split / "images" / image_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Downloaded image missing: {source_image}")

        image_destination = OUTPUT / split / "images" / image_name
        label_destination = OUTPUT / split / "labels" / f"{Path(image_name).stem}.txt"
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_image) as image:
            if image.size != (item["width"], item["height"]):
                raise ValueError(f"Unexpected source dimensions for {image_name}: {image.size}")
            image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS).save(
                image_destination, quality=95
            )

        segments = item.get("annotations", {}).get("segments", [])
        lines = [yolo_line(segment) for segment in segments]
        label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        image_counts[split] += 1
        segment_counts[split] += len(lines)

    yaml_lines = [
        f"path: {OUTPUT.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
        "",
    ]
    (OUTPUT / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
    summary = {
        "source_image_size": [640, 640],
        "output_image_size": list(SIZE),
        "classes": dict(enumerate(CLASS_NAMES)),
        "images_by_split": dict(image_counts),
        "segments_by_split": dict(segment_counts),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
