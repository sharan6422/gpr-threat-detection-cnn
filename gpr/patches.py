"""Cut fixed-size training patches out of annotated GPR B-scan images.

Input
-----
A folder of B-scan images and a JSON annotation file of the form::

    {"frames": {"scan_01.png": [{"x1": 10, "y1": 40, "x2": 42, "y2": 72}, ...], ...}}

Each box marks a buried threat.

Output
------
``<out>/images/*.png`` (grayscale, PATCH x PATCH) and ``<out>/patches.csv`` with
columns ``filename,label,source``. ``label`` is 1 for threat and 0 for non-threat.
``source`` is the B-scan the patch came from, so that training can split by
scan and keep near-duplicate patches out of the test set.

Usage::

    python -m gpr.patches --images data/scans --annotations data/labels.json --out data/patches
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

PATCH = 32
SHIFT = 4  # pixels per augmentation step


def _crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (PATCH, PATCH), interpolation=cv2.INTER_AREA)


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def threat_boxes(box: tuple[int, int, int, int], augment: bool) -> list[tuple[int, int, int, int]]:
    """The annotated box, plus shifted copies when augmenting.

    Shifts are SHIFT and 2*SHIFT pixels vertically and SHIFT pixels horizontally,
    following the shift-based augmentation described in Reichman et al.
    """
    x1, y1, x2, y2 = box
    boxes = [box]
    if augment:
        for dy in (-2 * SHIFT, -SHIFT, SHIFT, 2 * SHIFT):
            boxes.append((x1, y1 + dy, x2, y2 + dy))
        for dx in (-SHIFT, SHIFT):
            boxes.append((x1 + dx, y1, x2 + dx, y2))
    return boxes


def non_threat_boxes(
    shape: tuple[int, int], threats: list[tuple[int, int, int, int]], n: int, rng: random.Random
) -> list[tuple[int, int, int, int]]:
    """Random PATCH x PATCH windows that do not overlap any annotated threat."""
    h, w = shape
    boxes: list[tuple[int, int, int, int]] = []
    attempts = 0
    while len(boxes) < n and attempts < n * 50:
        attempts += 1
        x1 = rng.randint(0, w - PATCH)
        y1 = rng.randint(0, h - PATCH)
        box = (x1, y1, x1 + PATCH, y1 + PATCH)
        if not any(_overlaps(box, t) for t in threats):
            boxes.append(box)
    return boxes


def build(images: Path, annotations: Path, out: Path, augment: bool, neg_per_scan: int,
          exclude: list[str], seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    frames = json.loads(annotations.read_text())["frames"]
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int, str]] = []
    counts = {"scans": 0, "threat": 0, "non_threat": 0, "skipped_boxes": 0}
    for name, annots in sorted(frames.items()):
        if name in exclude:
            continue
        img = cv2.imread(str(images / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(images / name)
        counts["scans"] += 1
        stem = Path(name).stem
        threats = [(int(a["x1"]), int(a["y1"]), int(a["x2"]), int(a["y2"])) for a in annots]

        for i, t in enumerate(threats):
            for j, box in enumerate(threat_boxes(t, augment)):
                patch = _crop(img, box)
                if patch is None:
                    counts["skipped_boxes"] += 1
                    continue
                fname = f"{stem}_t{i}_{j}.png"
                cv2.imwrite(str(img_out / fname), patch)
                rows.append((fname, 1, name))
                counts["threat"] += 1

        for k, box in enumerate(non_threat_boxes(img.shape[:2], threats, neg_per_scan, rng)):
            fname = f"{stem}_n{k}.png"
            cv2.imwrite(str(img_out / fname), _crop(img, box))
            rows.append((fname, 0, name))
            counts["non_threat"] += 1

    with open(out / "patches.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "source"])
        writer.writerows(rows)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True, help="folder of B-scan images")
    p.add_argument("--annotations", type=Path, required=True, help="JSON file with threat boxes")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--augment", action="store_true", help="add shifted copies of each threat patch")
    p.add_argument("--neg-per-scan", type=int, default=27, help="non-threat patches sampled per scan")
    p.add_argument("--exclude", nargs="*", default=[], help="scan filenames to skip")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    print(build(a.images, a.annotations, a.out, a.augment, a.neg_per_scan, a.exclude, a.seed))


if __name__ == "__main__":
    main()
