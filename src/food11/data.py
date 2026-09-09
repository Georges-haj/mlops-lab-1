"""Prepare the Food-11 dataset for training.

Reads the raw Food-11 images (already split into training/evaluation/validation,
with the category encoded as the number before the first "_" in each filename,
e.g. "3_127.jpg" -> category index 3 -> "Egg") and produces two new datasets
organised the way an image classifier (e.g. ResNet, via
torchvision.datasets.ImageFolder) expects: one folder per category, inside
each split.

    <dest>/food11_processed/<split>/<category>/*.jpg
        All raw images, resized to <size>x<size> (default 128x128).

    <dest>/food11_processed_mini/<split>/<category>/*.jpg
        Same as above, capped at <mini_max> images per split/category
        (default 100). Meant for quick development so we can confirm the
        training code is correct before running on the full dataset.

Usage:
    uv run python ./src/food11/data.py
    uv run python ./src/food11/data.py --source-root data --dest-root data
"""

import argparse
from pathlib import Path

from PIL import Image

SPLITS = ["training", "evaluation", "validation"]

# Food-11: category index (the first part of the filename, before "_") -> name.
CATEGORIES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}

DEFAULT_SIZE = 128
DEFAULT_MINI_MAX = 100


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="data_local",
        help="Folder containing food11_raw/ (default: data_local)",
    )
    parser.add_argument(
        "--dest-root",
        default=None,
        help="Folder to write food11_processed(_mini)/ into (default: same as --source-root)",
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_SIZE, help="Output image side length in pixels"
    )
    parser.add_argument(
        "--mini-max",
        type=int,
        default=DEFAULT_MINI_MAX,
        help="Max images per split/category kept in the mini dataset",
    )
    return parser.parse_args()


def category_from_filename(filename: str) -> str:
    index_str = filename.split("_", 1)[0]
    try:
        index = int(index_str)
        return CATEGORIES[index]
    except (ValueError, KeyError) as exc:
        raise ValueError(f"can't determine category from filename {filename!r}") from exc


def process_split(
    raw_split_dir: Path,
    processed_split_dir: Path,
    mini_split_dir: Path,
    size: tuple[int, int],
    mini_max: int,
) -> tuple[int, dict[str, int]]:
    mini_counts: dict[str, int] = {}
    image_paths = sorted(p for p in raw_split_dir.iterdir() if p.is_file())
    processed_count = 0

    for image_path in image_paths:
        try:
            category = category_from_filename(image_path.name)
        except ValueError as exc:
            print(f"  [warn] {exc}, skipping")
            continue

        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize(size)

        out_dir = processed_split_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)
        resized.save(out_dir / image_path.name)
        processed_count += 1

        count_so_far = mini_counts.get(category, 0)
        if count_so_far < mini_max:
            mini_out_dir = mini_split_dir / category
            mini_out_dir.mkdir(parents=True, exist_ok=True)
            resized.save(mini_out_dir / image_path.name)
            mini_counts[category] = count_so_far + 1

    return processed_count, mini_counts


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root) if args.dest_root else source_root
    size = (args.size, args.size)

    raw_root = source_root / "food11_raw"
    processed_root = dest_root / "food11_processed"
    mini_root = dest_root / "food11_processed_mini"

    if not raw_root.exists():
        raise SystemExit(f"Raw data folder not found: {raw_root}")

    print(f"Reading raw images from:   {raw_root}")
    print(f"Writing food11_processed:  {processed_root}")
    print(f"Writing food11_processed_mini: {mini_root} (max {args.mini_max}/category/split)")

    for split in SPLITS:
        raw_split_dir = raw_root / split
        if not raw_split_dir.exists():
            print(f"  [skip] {split}: not found at {raw_split_dir}")
            continue

        total, mini_counts = process_split(
            raw_split_dir,
            processed_root / split,
            mini_root / split,
            size,
            args.mini_max,
        )
        mini_total = sum(mini_counts.values())
        print(f"  [{split}] {total} images processed, {mini_total} copied into mini")

    print("Done.")


if __name__ == "__main__":
    main()
