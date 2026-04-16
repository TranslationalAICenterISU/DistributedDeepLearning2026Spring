"""
Module 03 — DDP with Explicit Communication Primitives
-------------------------------------------------------
This module pulls back the curtain on what torch.nn.parallel.DDP
does automatically.  We do NOT use the DDP wrapper here.  Instead,
every inter-GPU communication is written out explicitly so you can
see exactly when data moves between ranks and why.

Primitives demonstrated:
  • dist.broadcast      — copy tensors from one rank to all others
  • dist.all_reduce     — sum (or average) a tensor across all ranks
  • dist.all_gather     — collect one tensor from every rank
  • dist.barrier        — global synchronisation point
  • dist.reduce         — sum to a single destination rank (rank 0)

How this maps to DDP internals:
  1. At startup  → broadcast(params, src=0) ensures identical init weights
  2. After loss.backward() → all_reduce(param.grad) averages gradients
  3. After validation      → all_reduce(metrics) gives global accuracy
"""
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm


# ── Model ─────────────────────────────────────────────────────────

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Communication helpers ─────────────────────────────────────────

def broadcast_parameters(model, src=0):
    """
    Copy every parameter tensor from rank `src` to all other ranks.
    This ensures all replicas start from the same random initialisation.
    DDP does this transparently; here we make it explicit.
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def sync_gradients(model, world_size):
    """
    Sum gradients across all ranks and divide by world_size so every
    rank ends up with the *average* gradient — exactly what DDP's
    gradient hook does during the backward pass.
    """
    for param in model.parameters():
        if param.grad is not None:
            # SUM across all ranks in place
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            # Divide to get the mean (equivalent to averaging mini-batches)
            param.grad.data /= world_size


def global_metric(local_loss, local_correct, local_n, device):
    """
    Aggregate scalar metrics across all ranks.
    Returns (global_avg_loss, global_accuracy) on every rank.
    """
    stats = torch.tensor([local_loss, float(local_correct), float(local_n)],
                         dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[2]).item(), (stats[1] / stats[2]).item()


def gather_per_rank_accuracy(local_acc, device, world_size):
    """
    Collect one float from every rank and print them on rank 0.
    Uses all_gather — every rank sends AND receives.
    Useful for inspecting load imbalance across ranks.
    """
    t = torch.tensor([local_acc], dtype=torch.float32, device=device)
    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    return [g.item() for g in gathered]


# ── Data ──────────────────────────────────────────────────────────

def get_dataloaders(data_dir, batch_size, rank, world_size, num_workers=4):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    if rank == 0:
        datasets.FashionMNIST(data_dir, train=True,  download=True)
        datasets.FashionMNIST(data_dir, train=False, download=True)
    dist.barrier()   # wait for rank 0 download before all ranks open the files

    train_ds = datasets.FashionMNIST(data_dir, train=True,  download=False, transform=train_tf)
    test_ds  = datasets.FashionMNIST(data_dir, train=False, download=False, transform=test_tf)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler  = DistributedSampler(test_ds,  num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, sampler=test_sampler,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, train_sampler


# ── Training / Evaluation ─────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, world_size, rank):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, desc="  train", leave=False, disable=(rank != 0))

    for X, y in pbar:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        # ── EXPLICIT GRADIENT SYNC ──────────────────────────────
        # This replaces the automatic gradient hook inside DDP.
        # Without this call, each rank would update with its own
        # local gradients only — models would immediately diverge.
        sync_gradients(model, world_size)
        # ───────────────────────────────────────────────────────

        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        n          += X.size(0)

    # Average loss / accuracy across all ranks
    return global_metric(total_loss, correct, n, device)


@torch.no_grad()
def evaluate(model, loader, criterion, device, world_size):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        total_loss += criterion(logits, y).item() * X.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        n          += X.size(0)

    return global_metric(total_loss, correct, n, device)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Module 03 — Explicit DDP Comms")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--data-dir",    default="/tmp/fashion_mnist")
    parser.add_argument("--save-dir",    default="./checkpoints_03")
    parser.add_argument("--num-workers", type=int,   default=4)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World size : {world_size}  (explicit comms mode)")
        print(f"Args       : {vars(args)}\n")
        os.makedirs(args.save_dir, exist_ok=True)
    dist.barrier()

    train_loader, test_loader, train_sampler = get_dataloaders(
        args.data_dir, args.batch_size, rank, world_size, args.num_workers
    )

    model     = FashionCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── EXPLICIT PARAMETER BROADCAST ────────────────────────────
    # Each rank initialises its own random weights.  We must broadcast
    # from rank 0 so every replica starts identically.
    # DDP does this in its __init__; here we do it ourselves.
    if rank == 0:
        print("Broadcasting initial parameters from rank 0 → all ranks...")
    broadcast_parameters(model, src=0)
    dist.barrier()
    if rank == 0:
        print("Broadcast complete.\n")
    # ────────────────────────────────────────────────────────────

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                            device, world_size, rank)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, world_size)
        scheduler.step()

        # ── ALL_GATHER: inspect per-rank validation accuracy ────
        per_rank_acc = gather_per_rank_accuracy(val_acc, device, world_size)
        # ────────────────────────────────────────────────────────

        if rank == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:2d}/{args.epochs}  |  "
                f"train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f}  acc {val_acc:.4f}  |  "
                f"{elapsed:.1f}s"
            )
            print(f"  per-rank val acc : {[f'{a:.4f}' for a in per_rank_acc]}")

            if val_acc > best_acc:
                best_acc = val_acc
                ckpt = os.path.join(args.save_dir, "best_model.pt")
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "val_acc": val_acc},
                    ckpt,
                )
                print(f"  ✓ Saved → {ckpt}")

    if rank == 0:
        print(f"\nBest validation accuracy : {best_acc:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
