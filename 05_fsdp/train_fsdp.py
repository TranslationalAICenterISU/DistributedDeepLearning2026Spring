"""
Module 05 — Fine-Tuning with Fully Sharded Data Parallel (FSDP)
----------------------------------------------------------------
FSDP is PyTorch's answer to training models that don't fit on a
single GPU.  Instead of each rank holding a full copy of all
parameters (as in DDP), FSDP shards parameters, gradients, and
optimizer states across all ranks.

  DDP  — each rank holds  1×  full model          (N×  memory)
  FSDP — each rank holds  1/N  of each parameter  (~1× memory total)

This script fine-tunes a pretrained ResNet-18 (from torchvision) on
Fashion MNIST to demonstrate:

  • How to wrap a model with FSDP
  • auto_wrap_policy  — automatically decide what to shard
  • MixedPrecision    — fp16 params/grads to halve memory & speed up comm
  • Proper checkpoint saving with FSDP (requires special state_dict APIs)

The same pattern applies to billion-parameter LLMs; ResNet-18 is
used here because it is available in torchvision without extra downloads.
"""
import os
import time
import argparse
import functools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Dataset / pretrained model
from torchvision import datasets, transforms, models


# ── Model ─────────────────────────────────────────────────────────

def build_model():
    """
    ResNet-18 pretrained on ImageNet, adapted for Fashion MNIST:
      • First conv layer: 3-channel → 1-channel  (grayscale input)
      • Final fc layer:   1000 classes → 10 classes
    """
    # weights=None avoids an internet download on compute nodes.
    # Swap to ResNet18_Weights.DEFAULT on a node with internet access
    # to start from ImageNet-pretrained weights instead.
    model = models.resnet18(weights=None)

    # Fashion MNIST is grayscale (1 channel), ImageNet is RGB (3 channels).
    # Replace the first conv to accept a single channel while keeping the
    # pretrained weights for the remaining layers.
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the classification head for 10 Fashion MNIST classes.
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


# ── Data ──────────────────────────────────────────────────────────

def get_dataloaders(data_dir, batch_size, rank, world_size, num_workers=4):
    # ResNet expects at least 32×32; resize Fashion MNIST from 28×28.
    train_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    if rank == 0:
        datasets.FashionMNIST(data_dir, train=True,  download=True)
        datasets.FashionMNIST(data_dir, train=False, download=True)
    dist.barrier()

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

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        n          += X.size(0)

    stats = torch.tensor([total_loss, float(correct), float(n)],
                         dtype=torch.float64, device=device)
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

    stats = torch.tensor([total_loss, float(correct), float(n)],
                         dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[2]).item(), (stats[1] / stats[2]).item()


# ── Checkpoint helpers ────────────────────────────────────────────

def save_fsdp_checkpoint(model, save_path, rank):
    """
    FSDP shards the model across ranks, so model.state_dict() by default
    returns only the local shard.  FULL_STATE_DICT gathers all shards to
    rank 0 first, giving us a complete, portable checkpoint.
    """
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state = model.state_dict()
    if rank == 0:
        torch.save(state, save_path)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Module 05 — FSDP Fine-Tuning")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--data-dir",    default="/tmp/fashion_mnist")
    parser.add_argument("--save-dir",    default="./checkpoints_05")
    parser.add_argument("--num-workers", type=int,   default=4)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World size : {world_size}  (FSDP mode)")
        print(f"Args       : {vars(args)}\n")
        os.makedirs(args.save_dir, exist_ok=True)
    dist.barrier()

    train_loader, test_loader, train_sampler = get_dataloaders(
        args.data_dir, args.batch_size, rank, world_size, args.num_workers
    )

    # ── Build and wrap with FSDP ────────────────────────────────
    model = build_model()

    # MixedPrecision: store params in fp16, communicate in fp16.
    # Reduces both GPU memory usage and communication volume by 2×.
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # auto_wrap_policy: FSDP decides which sub-modules to shard based on
    # parameter count.  Modules with ≥ min_num_params params get their own
    # FSDP unit; smaller ones are fused with their parent.
    wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e5)

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        device_id=local_rank,        # FSDP places the model on this GPU
    )

    if rank == 0:
        # Count only the parameters this rank is responsible for.
        local_params = sum(p.numel() for p in model.parameters())
        print(f"Local param count (rank 0 shard) : {local_params:,}\n")

    # ── Optimizer ───────────────────────────────────────────────
    # With FSDP, the optimizer sees only the local shard's parameters,
    # so memory for optimizer state is also sharded automatically.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if rank == 0:
            elapsed = time.time() - t0
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            print(
                f"Epoch {epoch:2d}/{args.epochs}  |  "
                f"train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f}  acc {val_acc:.4f}  |  "
                f"peak mem {mem:.2f} GB  |  {elapsed:.1f}s"
            )

        # val_acc is all_reduced so identical on every rank — safe to check on all.
        # FSDP.state_dict_type is a collective; every rank must enter together.
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = os.path.join(args.save_dir, "best_model.pt")
            save_fsdp_checkpoint(model, ckpt, rank)
            if rank == 0:
                print(f"  ✓ Saved FSDP checkpoint → {ckpt}  (val_acc={val_acc:.4f})")

    if rank == 0:
        print(f"\nBest validation accuracy : {best_acc:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
