#!/bin/bash
#SBATCH -J fashion-model-parallel
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH -p nova
#SBATCH -A short_term
#SBATCH -t 00:30:00
#SBATCH -o fashion_mp_%j.out
#SBATCH -e fashion_mp_%j.err
#SBATCH --gres=gpu:a100-pcie:2   # 2 GPUs on the SAME node

# Model parallelism splits the model across GPUs on one node.
# It is single-process: no srun / torchrun needed.

source config.sh
SAVE_DIR=${SAVE_BASE}/04_model_parallel
SCRIPT_DIR=04_model_parallel

DATA_DIR=$TMPDIR/fashion_mnist
mkdir -p $DATA_DIR $SAVE_DIR

echo "Node : $(hostname)"
echo "GPUs : $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "Date : $(date)"

$MICROMAMBA_BIN run -p $ENV_PREFIX \
    python $SCRIPT_DIR/train_mp.py  \
        --epochs     10             \
        --batch-size 256            \
        --lr         1e-3           \
        --data-dir   $DATA_DIR      \
        --save-dir   $SAVE_DIR
