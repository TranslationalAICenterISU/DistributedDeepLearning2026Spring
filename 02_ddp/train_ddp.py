"""
Module 02 — Simple Distributed Data Parallel (DDP)
----------------------------------------------------
Builds directly on Module 01.  The four changes needed to go from
single-GPU to multi-GPU DDP are marked with  ← DDP  comments:

  1. dist.init_process_group()          — join the distributed job
  2. torch.cuda.set_device(local_rank)  — pin each process to its GPU
  3. DistributedSampler                 — each rank sees a different shard
  4. DDP(model, device_ids=[local_rank])— automatic gradient all-reduce

Everything else (model, loss, optimizer, loop) is identical to Module 01.

Launch via:
  torchrun --nnodes=N --nproc-per-node=G ...  train_ddp.py
Or submit train_ddp.sh to Slurm.
"""
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist                          # ← DDP (1)
from torch.nn.parallel import DistributedDataParallel as DDP  # ← DDP (4)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # ← DDP (3)
from torchvision import datasets, transforms
from tqdm import tqdm


# ── Model (identical to Module 01) ───────────────────────────────

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


# ── Data ──────────────────────────────────────────────────────────

_MEAN = (0.2860,)
_STD  = (0.3530,)

def get_dataloaders(data_dir, batch_size, rank, world_size, num_workers=4):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    # Only rank 0 downloads; others wait for the files to appear.
    if rank == 0:
        datasets.FashionMNIST(data_dir, train=True,  download=True)
        datasets.FashionMNIST(data_dir, train=False, download=True)
    dist.barrier()

    train_ds = datasets.FashionMNIST(data_dir, train=True,  download=False, transform=train_tf)
    test_ds  = datasets.FashionMNIST(data_dir, train=False, download=False, transform=test_tf)

    # ← DDP (3): DistributedSampler partitions the dataset across ranks.
    #            shuffle=True inside the sampler; set_epoch() re-shuffles each epoch.
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler  = DistributedSampler(test_ds,  num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, sampler=test_sampler,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, train_sampler


# ── Training / Evaluation ─────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, desc="  train", leave=False, disable=(dist.get_rank() != 0))
    for X, y in pbar:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()           # DDP all-reduces gradients automatically here
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        n          += X.size(0)

    # Aggregate metrics across all ranks so every rank has the global numbers.
    stats = torch.tensor([total_loss, correct, n], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[2]).item(), (stats[1] / stats[2]).item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        total_loss += criterion(logits, y).item() * X.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        n          += X.size(0)

    stats = torch.tensor([total_loss, correct, n], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[2]).item(), (stats[1] / stats[2]).item()


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Module 02 — DDP Fashion MNIST")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--data-dir",    default="/tmp/fashion_mnist")
    parser.add_argument("--save-dir",    default="./checkpoints_02")
    parser.add_argument("--num-workers", type=int,   default=4)
    args = parser.parse_args()

    # ← DDP (1): initialise the process group.
    #   torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    #   automatically, so init_process_group picks them up via env://.
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # ← DDP (2): bind this process to its assigned GPU.
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World size : {world_size}")
        print(f"Args       : {vars(args)}\n")
        os.makedirs(args.save_dir, exist_ok=True)

    dist.barrier()

    # ── data ────────────────────────────────────────────────────
    train_loader, test_loader, train_sampler = get_dataloaders(
        args.data_dir, args.batch_size, rank, world_size, args.num_workers
    )

    # ── model + optimizer ────────────────────────────────────────
    model     = FashionCNN().to(device)
    # ← DDP (4): wrap model — DDP inserts gradient all-reduce hooks.
    model     = DDP(model, device_ids=[local_rank])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── training loop ────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # DistributedSampler must be told the epoch to get different shuffles.
        train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if rank == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:2d}/{args.epochs}  |  "
                f"train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f}  acc {val_acc:.4f}  |  "
                f"{elapsed:.1f}s"
            )
            if val_acc > best_acc:
                best_acc = val_acc
                ckpt = os.path.join(args.save_dir, "best_model.pt")
                # Save the inner module (unwrap DDP) so the checkpoint is portable.
                torch.save(
                    {"epoch": epoch, "model_state": model.module.state_dict(),
                     "val_acc": val_acc, "world_size": world_size},
                    ckpt,
                )
                print(f"  ✓ Saved best checkpoint → {ckpt}  (val_acc={val_acc:.4f})")

    if rank == 0:
        print(f"\nBest validation accuracy : {best_acc:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
