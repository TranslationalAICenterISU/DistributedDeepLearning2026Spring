#!/bin/bash
#SBATCH -J dist-demo
#SBATCH -N 2                      # 2 nodes
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --mem=32G                  # memory per node
#SBATCH -p nova
#SBATCH -A short_term
#SBATCH -t 00:05:00
#SBATCH -o dist_demo_%j.out
#SBATCH -e dist_demo_%j.err
#SBATCH --gres=gpu:a100-pcie:1

# micromamba bin + env prefix 
MICROMAMBA_BIN=/opt/rit/el9/20240413/app/linux-rhel9-x86_64_v3/gcc-11.3.1/micromamba-1.4.2-lcemqbetrry5tmxx3oygtoc3apv2af7e/bin/micromamba
ENV_PREFIX=/work/mech-ai/Aditya/Micromamba_demo

echo "Using micromamba from: $MICROMAMBA_BIN"
echo "Env prefix:           $ENV_PREFIX"

# Check
$MICROMAMBA_BIN --version

# dist config
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES:      $SLURM_NNODES"
echo "NTASKS:      $SLURM_NTASKS"

# launch: one srun task per node, torchrun handles rank assignment and rendezvous
srun $MICROMAMBA_BIN run -p $ENV_PREFIX \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=1 \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    /home/baditya/2025_DDP/test.py

