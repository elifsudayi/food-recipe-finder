"""Build the 10-class Food-101 subset with train / val / test splits.

- test  : the official Food-101 test images of the 10 classes (250 per class)
- train : official train images minus the validation images (650 per class)
- val   : VAL_PER_CLASS images per class, taken from the official train split
          with a fixed seed (100 per class)
The test set is never used for training or model selection.
"""
import csv
import random
import shutil
from pathlib import Path

from src.config import CLASSES, SEED

VAL_PER_CLASS = 100


def download_food101(root):
    """Download + extract Food-101 (~5 GB) with torchvision. Returns the food-101 folder."""
    from torchvision.datasets import Food101
    Food101(root=str(root), split="train", download=True)
    return Path(root) / "food-101"


def _read_ids(meta_file, classes):
    ids = {c: [] for c in classes}
    for line in Path(meta_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls = line.split("/")[0]
        if cls in ids:
            ids[cls].append(line)
    return {c: sorted(v) for c, v in ids.items()}


def build_subset(food101_dir, out_dir, manifest_path, classes=CLASSES,
                 val_per_class=VAL_PER_CLASS, seed=SEED):
    food101_dir, out_dir = Path(food101_dir), Path(out_dir)
    train_all = _read_ids(food101_dir / "meta" / "train.txt", classes)
    test_all = _read_ids(food101_dir / "meta" / "test.txt", classes)
    rng = random.Random(seed)

    rows = []
    for cls in classes:
        ids = list(train_all[cls])
        rng.shuffle(ids)
        split_ids = {
            "val": ids[:val_per_class],
            "train": ids[val_per_class:],
            "test": test_all[cls],
        }
        for split, items in split_ids.items():
            dst_dir = out_dir / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for item in items:
                src = food101_dir / "images" / f"{item}.jpg"
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                rows.append((split, cls, src.name))

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "filename"])
        w.writerows(rows)
    return rows
