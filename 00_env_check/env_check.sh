#!/bin/bash
#SBATCH -J env-check
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH -p nova
#SBATCH -A short_term
#SBATCH -t 00:05:00
#SBATCH -o env_check_%j.out
#SBATCH -e env_check_%j.err
#SBATCH --gres=gpu:a100-pcie:1

source config.sh
SCRIPT_DIR=00_env_check

echo "Node     : $(hostname)"
echo "User     : $USER"
echo "Date     : $(date)"
echo "Env      : $ENV_PREFIX"
echo ""

$MICROMAMBA_BIN run -p $ENV_PREFIX python $SCRIPT_DIR/env_check.py
