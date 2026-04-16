#!/bin/bash
#SBATCH -J fashion-fsdp
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH -p nova
#SBATCH -A short_term
#SBATCH -t 00:30:00
#SBATCH -o fashion_fsdp_%j.out
#SBATCH -e fashion_fsdp_%j.err
#SBATCH --gres=gpu:a100-pcie:1

source config.sh
SAVE_DIR=${SAVE_BASE}/05_fsdp
SCRIPT_DIR=05_fsdp

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
        $SCRIPT_DIR/train_fsdp.py       \
            --epochs     5              \
            --batch-size 128            \
            --lr         5e-4           \
            --data-dir   $DATA_DIR      \
            --save-dir   $SAVE_DIR
