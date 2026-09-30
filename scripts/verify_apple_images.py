"""Remove unreadable downloaded JPEGs so the resumable downloader can fetch them again."""

from pathlib import Path

from PIL import Image


ROOT = Path(r"D:\BIT\fruits\data\raw_apple_quality")
bad = []
for image_path in ROOT.glob("*\\images\\*.jpg"):
    try:
        with Image.open(image_path) as image:
            image.load()
    except Exception as exc:
        bad.append((image_path, str(exc)))

for image_path, _ in bad:
    image_path.unlink()

print(f"Removed {len(bad)} unreadable image(s).")
for image_path, reason in bad:
    print(f"{image_path}: {reason}")
