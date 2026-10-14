"""Pretrain GPRNet on grayscale CIFAR-10 so its early conv layers can be transferred to GPR.

CIFAR-10 images are already 32x32, the same size as the GPR patches, so no
resizing is needed; they are only converted to one channel.

Usage::

    python -m gpr.pretrain_cifar --data-dir data/cifar10 --epochs 10 --out runs/cifar_gray.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gpr.model import GPRNet

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    p.add_argument("--out", type=Path, default=Path("runs/cifar_gray.pt"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fake", action="store_true", help="use random FakeData instead of CIFAR-10 (smoke test)")
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.fake:
        ds = datasets.FakeData(size=256, image_size=(3, 32, 32), num_classes=10, transform=TRANSFORM)
    else:
        ds = datasets.CIFAR10(a.data_dir, train=True, download=True, transform=TRANSFORM)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True)

    model = GPRNet(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, a.epochs + 1):
        model.train()
        correct = seen = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss_fn(logits, y).backward()
            opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        print(f"epoch {epoch}  train_acc {correct / seen:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
