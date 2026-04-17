# Distributed Deep Learning Workshop
**Iowa State University — April 20, 2026**
**Instructor: Dr. Aditya Balu, Mech-AI Lab**

This repository contains seven self-contained modules that walk through GPU parallelism
strategies in PyTorch, from a single GPU up to fully sharded multi-node training and
LLM inference. Each module lives in its own folder with a Python script and a Slurm
batch submission script.

---

## Hardware

### Nova `nova` partition (default in scripts)
All scripts are pre-configured for Nova's `nova` partition with A100-PCIE-40 GB GPUs.

```bash
#SBATCH -p nova
#SBATCH -A short_term
#SBATCH --gres=gpu:a100-pcie:1        # or :2 for model parallelism
```

### Quadro RTX 6000 nodes (`nova20-frost-[3-7]`)
Five nodes each with 4 × Quadro RTX 6000 (24 GB GDDR6, Turing architecture).
Good for testing multi-GPU topologies without consuming A100 allocation.

```bash
# Single GPU
sbatch -A short_term --gres=gpu:rtx_6000:1 <script>.sh

# Two GPUs on one node (model parallelism)
sbatch -A short_term --gres=gpu:rtx_6000:2 04_model_parallel/train_mp.sh

# Multi-node (DDP / FSDP) — one RTX 6000 per node
sbatch -A short_term --gres=gpu:rtx_6000:1 02_ddp/train_ddp.sh
```

All seven modules were validated on these nodes (April 2026). Epochs run roughly
1.5–2× slower than A100-PCIE due to lower memory bandwidth, but accuracy is identical.

### DeepL reservation — `deepl` (nodes `nova21-gpu-[1-2,4-11]`, 4 × A100 per node)
A dedicated Slurm reservation for the workshop, active **April 20, 2026 07:00–17:00**.
These nodes carry 4 × A100 GPUs each. Account is `short_term`.

```bash
# Add --reservation=deepl to any sbatch command on April 20
sbatch -A short_term --reservation=deepl --gres=gpu:a100:1 01_single_gpu/train.sh

# Use all 4 GPUs on one node (single-node DDP or FSDP)
sbatch -A short_term --reservation=deepl \
    -N 1 --ntasks-per-node=4 --gres=gpu:a100:4 \
    02_ddp/train_ddp.sh
# (also set --nnodes=1 --nproc-per-node=4 in the torchrun line)
```

> The reservation overrides normal partition scheduling — jobs jump the queue and land
> on the reserved nodes directly. Outside the reservation window the flag is rejected.

---

## Switching Between GPU Topologies

The most common change you will make to the scripts is adjusting how many GPUs you
allocate and whether they span one node or many. The table below summarises the knobs.

### Same node, multiple GPUs (intra-node)

Use this for **model parallelism** (module 04) or when you want multiple DDP/FSDP
ranks without the overhead of inter-node networking.

```bash
#SBATCH -N 1                          # exactly 1 node
#SBATCH --ntasks-per-node=4           # 4 tasks → 4 ranks on that node
#SBATCH --gres=gpu:a100:4        # 4 GPUs allocated to that node
```

And in the `torchrun` call:
```bash
torchrun --nnodes=1 --nproc-per-node=4 ...
```

Intra-node communication goes over NVLink (SXM) or PCIe (PCIE variant), which is
significantly faster than InfiniBand. Prefer this topology when the model fits across
the GPUs on one node.

### Multiple nodes, one GPU each (inter-node)

Used by modules 02, 03, and 05 in their default configuration.

```bash
#SBATCH -N 2                          # 2 nodes
#SBATCH --ntasks-per-node=1           # 1 task (rank) per node
#SBATCH --gres=gpu:a100:1        # 1 GPU per node
```

```bash
torchrun --nnodes=2 --nproc-per-node=1 ...
```

### Multiple nodes, multiple GPUs each (full scale)

Combine both: e.g. 2 nodes × 4 GPUs = world size 8.

```bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
```

```bash
torchrun --nnodes=2 --nproc-per-node=4 \
    --rdzv-id=$SLURM_JOB_ID           \
    --rdzv-backend=c10d               \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_ddp.py ...
```

The `MASTER_ADDR` and `MASTER_PORT` variables are already set in the multi-node
scripts; you only need to adjust `--nnodes` and `--nproc-per-node`.

---

## One-time Setup

### 1. Clone / copy this repository
```bash
cd /work/short_term/$USER
git clone <repo-url> 2025_DDP
cd 2025_DDP
```

### 2. Edit `config.sh`
This is the only file that needs your site-specific paths. Open it and set:

| Variable | Instructor value | Participant value |
|---|---|---|
| `GROUP_DIR` | `/work/mech-ai/Aditya` | `/work/short_term` |
| `ENV_PREFIX` | `${GROUP_DIR}/Micromamba_demo` | `${GROUP_DIR}/$USER/mamba_envs/dl_workshop` |
| `DATA_DIR` | `${GROUP_DIR}/data` | `${GROUP_DIR}/$USER/data` |
| `SAVE_BASE` | `${GROUP_DIR}/checkpoints` | `${GROUP_DIR}/$USER/checkpoints` |

```bash
# config.sh excerpt — edit these four lines
GROUP_DIR=/work/short_term          # your working directory
ENV_PREFIX=${GROUP_DIR}/$USER/mamba_envs/dl_workshop
DATA_DIR=${GROUP_DIR}/$USER/data
SAVE_BASE=${GROUP_DIR}/$USER/checkpoints
```

### 3. Create the micromamba environment (first time only)
```bash
module load micromamba
micromamba create -p $ENV_PREFIX python=3.11 -y
micromamba run -p $ENV_PREFIX pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
micromamba run -p $ENV_PREFIX pip install \
    tqdm matplotlib scikit-learn ollama
```

### 4. Verify the environment
```bash
sbatch 00_env_check/env_check.sh
# check: env_check_<jobid>.out  — should end with all PASS lines
```

---

## Module Reference

All scripts are submitted from the repository root:
```bash
sbatch <module>/script.sh
```

### 00 — Environment Check
**Script:** `00_env_check/env_check.py` | **Slurm:** `00_env_check/env_check.sh`

Verifies Python, PyTorch, CUDA, NCCL/Gloo, torchvision, FSDP imports, and the ollama
client. Runs a 2-rank Gloo smoke test using multiprocessing. No GPU required.

```
#SBATCH -p nova               # CPU-only job is fine
```

---

### 01 — Single GPU Training
**Script:** `01_single_gpu/train.py` | **Slurm:** `01_single_gpu/train.sh`

Trains a small CNN on Fashion MNIST (28×28 grayscale, 10 classes) on one GPU.
Establishes the baseline code structure that all later modules build on.

Key components: `FashionCNN`, `AdamW`, `CosineAnnealingLR`, tqdm progress bars,
best-checkpoint saving.

**Resources:** 1 GPU, ~30 min for 10 epochs
**Typical result:** ~91–92 % validation accuracy

---

### 02 — Data Parallel (DDP)
**Script:** `02_ddp/train_ddp.py` | **Slurm:** `02_ddp/train_ddp.sh`

Wraps the module-01 model with `torch.nn.parallel.DistributedDataParallel`.
Each rank holds a full copy of the model but sees a disjoint shard of the data.
Gradients are all-reduced automatically by the DDP wrapper.

Four lines changed from module 01 (marked `← DDP` in the source):
1. `dist.init_process_group("nccl")`
2. `torch.cuda.set_device(local_rank)`
3. `DistributedSampler` on the DataLoader
4. `DDP(model, device_ids=[local_rank])`

**Default resources:** 2 nodes × 1 GPU (world size 2)
**To scale:** increase `-N` and `--nnodes` together; `world_size` is inferred from
`SLURM_NTASKS`.

---

### 03 — Explicit Distributed Communications
**Script:** `03_ddp_explicit_comms/train_ddp_explicit.py` | **Slurm:** `03_ddp_explicit_comms/train_ddp_explicit.sh`

Same model and data as module 02 but **no DDP wrapper**. Instead the script calls
the communication primitives directly, so you can see exactly what DDP does under
the hood:

| Primitive | Used for |
|---|---|
| `dist.broadcast` | Sync initial model weights from rank 0 to all |
| `dist.all_reduce` | Average gradients after each backward pass |
| `dist.all_gather` | Collect per-rank validation accuracy |
| `dist.barrier` | Ensure all ranks reach the same point |

**Default resources:** 2 nodes × 1 GPU

---

### 04 — Model Parallelism
**Script:** `04_model_parallel/train_mp.py` | **Slurm:** `04_model_parallel/train_mp.sh`

Splits a single model across **two GPUs on the same node**. The feature-extraction
layers live on `cuda:0`; the classifier head lives on `cuda:1`. The forward pass
moves activations between devices with `.to(dev1)`.

This is a single-process job — no `srun` or `torchrun`. Useful when a model is too
large for one GPU but fits across two.

**Resources:** 1 node, **2 GPUs** (same node required)

```bash
# In train_mp.sh — change GPU count only, node count stays 1
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2        # must be on the same node
```

Note: `DATA_DIR` uses `$TMPDIR` (node-local scratch) to avoid file-corruption when
multiple jobs download the dataset concurrently.

---

### 05 — FSDP Fine-Tuning
**Script:** `05_fsdp/train_fsdp.py` | **Slurm:** `05_fsdp/train_fsdp.sh`

Fine-tunes a ResNet-18 on Fashion MNIST with **Fully Sharded Data Parallel**.
Unlike DDP where every rank holds the full model, FSDP shards parameters,
gradients, and optimizer states across ranks — peak per-GPU memory scales as
roughly `1/world_size`.

Key concepts demonstrated:

- `MixedPrecision(param_dtype=torch.float16, ...)` — halves memory and communication
- `size_based_auto_wrap_policy(min_num_params=1e5)` — auto-shard sub-modules
- `FSDP.state_dict_type(FULL_STATE_DICT)` — gather shards to rank 0 for checkpointing

> **Important:** `FSDP.state_dict_type` is a collective — every rank must call it
> together. The checkpoint call is therefore placed outside any `if rank == 0:` guard.

**Default resources:** 2 nodes × 1 GPU

To use 4 GPUs on a single `ach` node instead:
```bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
# and in the torchrun line:
torchrun --nnodes=1 --nproc-per-node=4 ...
```

---

### 06 — Ollama LLM Inference
**Script:** `06_ollama/inference.py` | **Slurm:** `06_ollama/inference.sh`

Demonstrates local LLM inference using [Ollama](https://ollama.com) on a single GPU.
The batch script starts the Ollama server, pulls the model if not cached, runs six
demo scenarios, then stops the server cleanly.

Demo scenarios:
1. List available models
2. Basic single-turn generation
3. Streaming token-by-token output
4. System prompt / persona
5. Multi-turn conversation
6. Parallel batch queries

**Resources:** 1 GPU per run (models run sequentially; 40 GB A100 fits each workshop model)
**Model store:** `$OLLAMA_MODELS` (set in `config.sh` — pre-populate with `pull_models.sh`)

Demo sections:

| Section | Model | Focus |
|---|---|---|
| 1 | `gpt-oss:20b` | Chain-of-thought reasoning, agentic task planning |
| 2 | `qwen3.6` | Code generation, streaming explanation, bug identification |
| 3 | `gemma4` | Multimodal: describe + classify a Fashion MNIST image |

Each section ends with a GPU placement check (`ollama ps` + `nvidia-smi`) that confirms
the model loaded into VRAM rather than falling back to CPU.

#### Running a large model that needs the full node

Some models (70B+) exceed a single GPU's memory. Ollama automatically spreads across
all GPUs visible on the node — you just need to allocate them all in Slurm.

**Option A — single node, all 4 GPUs (A100, `deepl` reservation):**

```bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH -p nova
#SBATCH -A short_term
#SBATCH --reservation=deepl        # workshop reservation, April 20 only
#SBATCH --gres=gpu:a100:4          # give Ollama all 4 GPUs (4 × 40 GB = 160 GB total)
#SBATCH -t 00:30:00
```

Ollama sees all four GPUs via `CUDA_VISIBLE_DEVICES` and shards the model weights
across them automatically — no code changes needed.

**Option B — single node, all 8 RTX Pro 6000 (`nova26-gpu-1`):**

```bash
#SBATCH --nodelist=nova26-gpu-1
#SBATCH --gres=gpu:rtx_pro_6000:8  # 8 × 48 GB = 384 GB total VRAM
```

**Rough VRAM requirements by model size (Q4 quantisation):**

| Parameters | VRAM needed | Fits on |
|---|---|---|
| 7 B | ~5 GB | any single A100 |
| 20 B | ~12 GB | any single A100 |
| 70 B | ~40 GB | 1 × A100-80G or 2 × A100-40G |
| 405 B | ~230 GB | 6 × A100-40G or 3 × A100-80G |

---

## Results Summary (tested on Nova, April 2026)

### A100-PCIE-40 GB (`nova` partition)

| Module | Hardware | Epochs | Val Acc | Time/epoch |
|---|---|---|---|---|
| 01 single GPU | 1 × A100-PCIE-40G | 10 | 91.5 % | ~16 s |
| 02 DDP | 2 × A100-PCIE-40G (2 nodes) | 10 | 91.8 % | ~16 s |
| 03 DDP explicit | 2 × A100-PCIE-40G (2 nodes) | 10 | 91.7 % | ~17 s |
| 04 model parallel | 2 × A100-PCIE-40G (1 node) | 10 | 92.0 % | ~17 s |
| 05 FSDP | 2 × A100-PCIE-40G (2 nodes) | 5 | 90.6 % | ~33 s |
| 06 Ollama | 1 × A100-PCIE-40G | — | llama3.2 | ~8 s/query |

### Quadro RTX 6000-24 GB (`nova20-frost` nodes, `-A short_term`)

| Module | Hardware | Epochs | Val Acc | Time/epoch |
|---|---|---|---|---|
| 01 single GPU | 1 × RTX 6000 | 10 | ~91 % | ~31 s |
| 02 DDP | 2 × RTX 6000 (2 nodes) | 10 | ~91 % | ~24 s |
| 03 DDP explicit | 2 × RTX 6000 (2 nodes) | 10 | ~91 % | ~22 s |
| 04 model parallel | 2 × RTX 6000 (1 node) | 10 | ~91 % | ~26 s |
| 05 FSDP | 2 × RTX 6000 (2 nodes) | 5 | ~90 % | ~46 s |
| 06 Ollama | 1 × RTX 6000 | — | llama3.2 | — |

FSDP trains ResNet-18 (vs. the lightweight CNN in 01–04), hence lower accuracy in
fewer epochs — the point is memory efficiency, not peak accuracy.

---

## Common Slurm Tips

```bash
# Submit a job
sbatch <module>/script.sh

# Check job status
squeue -u $USER

# Cancel a job
scancel <jobid>

# Tail live output
tail -f fashion_ddp_<jobid>.out

# Interactive GPU session on nova
salloc -N 1 --gres=gpu:a100-pcie:1 -p nova -A short_term -t 01:00:00

# Interactive GPU session on ach (DeepL reservation)
salloc -N 1 --gres=gpu:a100:1 -p ach -A deepl -t 01:00:00
```

---

## Repository Layout

```
2025_DDP/
├── config.sh                        # central path / env config — edit this first
├── 00_env_check/
│   ├── env_check.py
│   └── env_check.sh
├── 01_single_gpu/
│   ├── train.py
│   └── train.sh
├── 02_ddp/
│   ├── train_ddp.py
│   └── train_ddp.sh
├── 03_ddp_explicit_comms/
│   ├── train_ddp_explicit.py
│   └── train_ddp_explicit.sh
├── 04_model_parallel/
│   ├── train_mp.py
│   └── train_mp.sh
├── 05_fsdp/
│   ├── train_fsdp.py
│   └── train_fsdp.sh
└── 06_ollama/
    ├── inference.py
    └── inference.sh
```
