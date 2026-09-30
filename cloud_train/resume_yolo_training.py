"""Resume an interrupted Ultralytics training run from its last checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    YOLO(str(args.checkpoint)).train(resume=True)


if __name__ == "__main__":
    main()
