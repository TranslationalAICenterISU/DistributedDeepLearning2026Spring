import os
import torch
import torch.distributed as dist

def main():
    # Initialize process group using env variables:
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # One gpu per node in your setup for now
    device = torch.device("cuda:0")

    # Simple tensor addition to the rank of the gpus
    x = torch.tensor([rank + 1.0], device=device)

    # All-reduce sum across all ranks
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # Print from every rank 
    print(f"[Rank {rank}/{world_size}] tensor after all_reduce = {x.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    main()

