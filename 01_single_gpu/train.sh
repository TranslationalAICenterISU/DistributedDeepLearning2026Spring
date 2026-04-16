#!/bin/bash
#SBATCH -J fashion-single
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH -p nova
#SBATCH -A short_term
#SBATCH -t 00:30:00
#SBATCH -o fashion_single_%j.out
#SBATCH -e fashion_single_%j.err
#SBATCH --gres=gpu:a100-pcie:1

source config.sh
SAVE_DIR=${SAVE_BASE}/01_single_gpu
SCRIPT_DIR=01_single_gpu

mkdir -p $DATA_DIR $SAVE_DIR

echo "Node     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date     : $(date)"

$MICROMAMBA_BIN run -p $ENV_PREFIX \
    python $SCRIPT_DIR/train.py \
        --epochs     10          \
        --batch-size 256         \
        --lr         1e-3        \
        --data-dir   $DATA_DIR   \
        --save-dir   $SAVE_DIR
