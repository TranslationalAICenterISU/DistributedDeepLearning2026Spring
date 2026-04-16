"""
Module 01 — Single-GPU Training: Fashion MNIST
-----------------------------------------------
Full training pipeline on one GPU:
  • Data loading with augmentation
  • CNN model definition
  • Training loop with progress bars
  • Validation after every epoch
  • Cosine LR schedule
  • Best-checkpoint saving

This is the baseline that all subsequent distributed modules build on.
"""
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ── Model ─────────────────────────────────────────────────────────

class FashionCNN(nn.Module):
    """Small CNN: two conv blocks followed by an MLP head."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1×28×28 → 32×14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32×14×14 → 64×7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Data ──────────────────────────────────────────────────────────

# Fashion MNIST channel mean / std
_MEAN = (0.2860,)
_STD  = (0.3530,)

def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_ds = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ── Training / Evaluation ─────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in tqdm(loader, desc="  train", leave=False):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        n          += X.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in tqdm(loader, desc="  eval ", leave=False):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        total_loss += criterion(logits, y).item() * X.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        n          += X.size(0)

    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Module 01 — Single-GPU Fashion MNIST")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--data-dir",    default="/tmp/fashion_mnist")
    parser.add_argument("--save-dir",    default="./checkpoints_01")
    parser.add_argument("--num-workers", type=int,   default=4)
    args = parser.parse_args()

    # ── device ──────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Request --gres=gpu in your Slurm script.")
    device = torch.device("cuda:0")
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Args   : {vars(args)}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── data ────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Train batches : {len(train_loader)}  "
          f"({len(train_loader.dataset)} samples)")
    print(f"Test  batches : {len(test_loader)}  "
          f"({len(test_loader.dataset)} samples)\n")

    # ── model + optimizer ────────────────────────────────────────
    model     = FashionCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params  : {total_params:,}\n")

    # ── training loop ────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:2d}/{args.epochs}  |  "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}  |  "
            f"lr {scheduler.get_last_lr()[0]:.2e}  |  "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = os.path.join(args.save_dir, "best_model.pt")
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_acc": val_acc, "args": vars(args)},
                ckpt,
            )
            print(f"  ✓ Saved best checkpoint → {ckpt}  (val_acc={val_acc:.4f})")

    print(f"\nBest validation accuracy : {best_acc:.4f}")


if __name__ == "__main__":
    main()
