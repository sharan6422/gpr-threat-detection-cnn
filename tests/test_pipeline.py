"""End-to-end smoke test on synthetic B-scans: patches -> pretrain -> transfer -> train -> metrics."""
import json

import cv2
import numpy as np
import torch

from gpr import pretrain_cifar
from gpr.model import GPRNet, transfer_conv_layers
from gpr.patches import build
from gpr.train import parse_args, run


def _synthetic_scans(root, n_scans=12, seed=0):
    """Noisy B-scans with a bright hyperbola wherever a 'threat' is annotated."""
    rng = np.random.default_rng(seed)
    scans = root / "scans"
    scans.mkdir()
    frames = {}
    for s in range(n_scans):
        img = (rng.normal(100, 20, (128, 256))).clip(0, 255).astype(np.uint8)
        boxes = []
        for _ in range(2):
            cx, cy = int(rng.integers(40, 216)), int(rng.integers(30, 90))
            xs = np.arange(cx - 16, cx + 16)
            ys = (cy + 0.02 * (xs - cx) ** 2).astype(int)
            for x, y in zip(xs, ys):
                cv2.circle(img, (int(x), int(y)), 2, 255, -1)
            boxes.append({"x1": cx - 16, "y1": cy - 4, "x2": cx + 16, "y2": cy + 28})
        name = f"scan_{s:02d}.png"
        cv2.imwrite(str(scans / name), img)
        frames[name] = boxes
    ann = root / "labels.json"
    ann.write_text(json.dumps({"frames": frames}))
    return scans, ann


def test_transfer_copies_weights():
    src, dst = GPRNet(num_classes=10), GPRNet()
    n = transfer_conv_layers(dst, src.state_dict(), 2)
    assert n == 2
    assert torch.equal(dst.features[0].weight, src.features[0].weight)
    assert torch.equal(dst.features[3].weight, src.features[3].weight)
    assert not torch.equal(dst.features[7].weight, src.features[7].weight)


def test_end_to_end(tmp_path):
    scans, ann = _synthetic_scans(tmp_path)
    counts = build(scans, ann, tmp_path / "patches", augment=True, neg_per_scan=20, exclude=[], seed=1)
    assert counts["threat"] > 0 and counts["non_threat"] == 12 * 20

    pre = tmp_path / "cifar.pt"
    pretrain_cifar.main(["--fake", "--epochs", "1", "--out", str(pre)])

    args = parse_args([
        "--csv", str(tmp_path / "patches" / "patches.csv"),
        "--img-dir", str(tmp_path / "patches" / "images"),
        "--out", str(tmp_path / "run"), "--pretrained", str(pre),
        "--epochs", "8", "--val-frac", "0.25", "--test-frac", "0.25",
    ])
    result = run(args)

    # No scan appears in more than one split.
    import pandas as pd
    from gpr.data import split_by_scan
    tr, va, te = split_by_scan(pd.read_csv(args.csv), 0.25, 0.25, 42)
    assert not (set(tr.source) & set(te.source)) and not (set(va.source) & set(te.source))

    assert (tmp_path / "run" / "best.pt").exists()
    # The synthetic task is easy, so a working pipeline should clearly beat chance,
    # and must actually flag threats at its chosen threshold (not just rank them).
    test = result["test"]
    assert test["roc_auc"] > 0.8
    assert test["recall"] > 0.7 and test["precision"] > 0.7
