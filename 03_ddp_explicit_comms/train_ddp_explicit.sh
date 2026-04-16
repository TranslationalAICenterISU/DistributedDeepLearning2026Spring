#!/bin/bash
#SBATCH -J fashion-ddp-explicit
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH -p nova
#SBATCH -A short_term
#SBATCH -t 00:30:00
#SBATCH -o fashion_ddp_explicit_%j.out
#SBATCH -e fashion_ddp_explicit_%j.err
#SBATCH --gres=gpu:a100-pcie:1

source config.sh
SAVE_DIR=${SAVE_BASE}/03_ddp_explicit
SCRIPT_DIR=03_ddp_explicit_comms

mkdir -p $DATA_DIR $SAVE_DIR

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Nodes      : $SLURM_NNODES  ($SLURM_JOB_NODELIST)"
echo "MASTER     : $MASTER_ADDR:$MASTER_PORT"
echo "World size : $SLURM_NTASKS"
echo "Date       : $(date)"

srun $MICROMAMBA_BIN run -p $ENV_PREFIX \
    torchrun \
        --nnodes=$SLURM_NNODES          \
        --nproc-per-node=1              \
        --rdzv-id=$SLURM_JOB_ID         \
        --rdzv-backend=c10d             \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        $SCRIPT_DIR/train_ddp_explicit.py \
            --epochs     10             \
            --batch-size 256            \
            --lr         1e-3           \
            --data-dir   $DATA_DIR      \
            --save-dir   $SAVE_DIR
