#!/bin/bash
#SBATCH -J ollama-inference
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH -p nova
#SBATCH -A short_term
#SBATCH -t 00:30:00
#SBATCH -o ollama_inference_%j.out
#SBATCH -e ollama_inference_%j.err
#SBATCH --gres=gpu:a100-pcie:1

source config.sh
SCRIPT_DIR=06_ollama

# Model to use for the demo — change this to any pulled model.
# Smaller models (llama3.2 3B) respond faster; larger ones are more capable.
MODEL=llama3.2

echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date         : $(date)"
echo "Ollama bin   : $OLLAMA_BIN"
echo "Model store  : $OLLAMA_MODELS"
echo "Model        : $MODEL"
echo ""

# ── Start Ollama server ────────────────────────────────────────────
# Run in the background; capture its PID so we can stop it cleanly.
$OLLAMA_BIN serve &
OLLAMA_PID=$!
echo "Started ollama serve (PID $OLLAMA_PID)"

# Give the server a moment to bind the port and load GPU drivers.
sleep 8

# ── Pull the model if not already cached ──────────────────────────
echo "Pulling model $MODEL (skipped if already cached)..."
$OLLAMA_BIN pull $MODEL

# ── Run inference demos ───────────────────────────────────────────
$MICROMAMBA_BIN run -p $ENV_PREFIX \
    python $SCRIPT_DIR/inference.py \
        --model $MODEL              \
        --host  http://localhost:11434

# ── Clean up ──────────────────────────────────────────────────────
echo ""
echo "Stopping ollama server (PID $OLLAMA_PID)..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null
echo "Done."
