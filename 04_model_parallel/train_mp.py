"""
Module 04 — Model Parallelism
------------------------------
Model Parallelism splits the model itself across multiple devices,
rather than replicating it.  It is used when a single model is too
large to fit in one GPU's memory.

This script demonstrates a two-stage pipeline on two GPUs (cuda:0
and cuda:1) on the same node:

  GPU 0 — feature extractor  (conv layers)
  GPU 1 — classifier head    (linear layers)

During the forward pass, activations are transferred from GPU 0 to
GPU 1 via PCIe / NVLink.  Gradients flow back across the same link
during the backward pass.

NOTE: This requires a single node with at least 2 GPUs.
      Request --gres=gpu:a100-pcie:2 in your Slurm script.

Compare with Module 01:
  • Same dataset and training loop.
  • The only difference is how the model is placed on devices.
  • No dist.init_process_group needed — this is intra-node only.
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


# ── Model split across two GPUs ───────────────────────────────────

class ModelParallelCNN(nn.Module):
    """
    Identical architecture to FashionCNN in Module 01, but:
      self.stage0  lives on  cuda:0  (feature extraction)
      self.stage1  lives on  cuda:1  (classification)

    The .to() calls in __init__ permanently assign each sub-module to
    its device.  PyTorch tracks which device each parameter is on and
    computes gradients correctly across the device boundary.
    """

    def __init__(self, dev0="cuda:0", dev1="cuda:1"):
        super().__init__()
        self.dev0 = dev0
        self.dev1 = dev1

        # ── Stage 0: convolutional feature extractor  (GPU 0) ──
        self.stage0 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ).to(dev0)

        # ── Stage 1: MLP classifier  (GPU 1) ───────────────────
        self.stage1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 10),
        ).to(dev1)

    def forward(self, x):
        # Input arrives on CPU (DataLoader default).
        # Move to stage 0 device, run features.
        x = self.stage0(x.to(self.dev0))

        # Transfer intermediate activations to stage 1 device.
        # This .to() call is the cross-GPU communication.
        x = self.stage1(x.to(self.dev1))
        return x   # logits live on dev1


# ── Data ──────────────────────────────────────────────────────────

def get_dataloaders(data_dir, batch_size, num_workers=4):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader


# ── Training / Evaluation ─────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, label_device):
    """label_device: device where logits and labels must meet for the loss."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in tqdm(loader, desc="  train", leave=False):
        # Labels must be on the same device as the model's final output.
        y = y.to(label_device)
        optimizer.zero_grad()
        logits = model(X)           # X starts on CPU; model moves it internally
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        n          += X.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, label_device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        y = y.to(label_device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * X.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        n          += X.size(0)
    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Module 04 — Model Parallelism")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--data-dir",    default="/tmp/fashion_mnist")
    parser.add_argument("--save-dir",    default="./checkpoints_04")
    parser.add_argument("--num-workers", type=int,   default=4)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(
            f"Model parallelism requires ≥ 2 GPUs on this node.  "
            f"Found {n_gpus}.  Request --gres=gpu:a100-pcie:2 in Slurm."
        )

    dev0, dev1 = "cuda:0", "cuda:1"
    print(f"GPU 0 (stage 0 / features)   : {torch.cuda.get_device_name(0)}")
    print(f"GPU 1 (stage 1 / classifier) : {torch.cuda.get_device_name(1)}")
    print(f"Args : {vars(args)}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    model = ModelParallelCNN(dev0, dev1)

    # Optimizer must cover parameters on both GPUs.
    # AdamW handles mixed-device parameter groups automatically.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Criterion operates on dev1 since that is where logits are produced.
    criterion = nn.CrossEntropyLoss().to(dev1)

    # Print memory usage per GPU before training
    for i in range(n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e6
        print(f"  GPU {i} memory allocated (before training): {alloc:.1f} MB")
    print()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, dev1)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion, dev1)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:2d}/{args.epochs}  |  "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}  |  "
            f"{elapsed:.1f}s"
        )

        # Show live GPU memory split between the two devices
        mem0 = torch.cuda.memory_allocated(0) / 1e6
        mem1 = torch.cuda.memory_allocated(1) / 1e6
        print(f"  GPU 0 mem: {mem0:.1f} MB  |  GPU 1 mem: {mem1:.1f} MB")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = os.path.join(args.save_dir, "best_model.pt")
            # Move everything to CPU before saving so the checkpoint is portable.
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state": cpu_state, "val_acc": val_acc}, ckpt)
            print(f"  ✓ Saved → {ckpt}")

    print(f"\nBest validation accuracy : {best_acc:.4f}")


if __name__ == "__main__":
    main()
