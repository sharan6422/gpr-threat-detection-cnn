"""Train GPRNet on patches, select the best epoch on validation, report on a held-out test set.

Usage::

    python -m gpr.train --csv data/patches/patches.csv --img-dir data/patches/images \
        --pretrained runs/cifar_gray.pt --transfer-layers 4 --out runs/gpr
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from gpr.data import PatchDataset, split_by_scan
from gpr.model import GPRNet, transfer_conv_layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            criterion: nn.Module | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Threat probabilities, labels and (if a criterion is given) mean loss."""
    model.eval()
    probs, labels, total = [], [], 0.0
    for x, y in loader:
        logits = model(x.to(device))
        if criterion is not None:
            total += criterion(logits, y.to(device)).item() * len(y)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    labels_arr = np.concatenate(labels)
    return np.concatenate(probs), labels_arr, total / max(len(labels_arr), 1)


def pick_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising Youden's J (TPR - FPR) on validation data."""
    if len(set(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))


def metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    out = {
        "accuracy": float((preds == labels).mean()),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "n": int(len(labels)), "n_threat": int(labels.sum()),
    }
    out["roc_auc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None
    return out


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    train_df, val_df, test_df = split_by_scan(df, args.val_frac, args.test_frac, args.seed)
    split_info = {name: {"patches": len(d), "scans": d["source"].nunique(), "threat": int(d["label"].sum())}
                  for name, d in (("train", train_df), ("val", val_df), ("test", test_df))}
    print("split:", split_info)

    def loader(d: pd.DataFrame, train: bool) -> DataLoader:
        return DataLoader(PatchDataset(d, args.img_dir, train=train), batch_size=args.batch_size, shuffle=train)

    train_dl, val_dl, test_dl = loader(train_df, True), loader(val_df, False), loader(test_df, False)

    model = GPRNet().to(device)
    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        n = transfer_conv_layers(model, state, args.transfer_layers)
        print(f"transferred {n} conv layers from {args.pretrained}")

    # Non-threat patches usually outnumber threats, so weight the loss by inverse class frequency.
    counts = np.bincount(train_df["label"].astype(int), minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Best epoch = highest validation AUC, ties broken by lower validation loss.
    # AUC alone ignores calibration: an early epoch can rank perfectly while every
    # probability is below 0.5, so loss is needed to separate tied epochs.
    history, best_key, best_epoch = [], (-1.0, float("-inf")), -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            seen += len(y)
        val_probs, val_labels, val_loss = predict(model, val_dl, device, criterion)
        val = metrics(val_probs, val_labels)
        auc = val["roc_auc"] if val["roc_auc"] is not None else val["accuracy"]
        history.append({"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss,
                        "val_auc": val["roc_auc"], "val_acc": val["accuracy"]})
        print(f"epoch {epoch:3d}  train_loss {total / seen:.4f}  val_loss {val_loss:.4f}  "
              f"val_auc {auc:.4f}  val_acc {val['accuracy']:.3f}")
        if (auc, -val_loss) > best_key:
            best_key, best_epoch = (auc, -val_loss), epoch
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    val_probs, val_labels, _ = predict(model, val_dl, device)
    threshold = pick_threshold(val_probs, val_labels)
    test_probs, test_labels, _ = predict(model, test_dl, device)
    result = {
        "best_epoch": best_epoch,
        "threshold": threshold,
        "split": split_info,
        "test": metrics(test_probs, test_labels, threshold),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print("test:", json.dumps(result["test"], indent=2))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/gpr"))
    p.add_argument("--pretrained", type=Path, help="state dict from gpr.pretrain_cifar")
    p.add_argument("--transfer-layers", type=int, default=4, help="conv layers to copy from the pretrained net (0-6)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
