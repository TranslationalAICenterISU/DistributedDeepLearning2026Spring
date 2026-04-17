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

echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date         : $(date)"
echo "Ollama bin   : $OLLAMA_BIN"
echo "Model store  : $OLLAMA_MODELS"
echo ""

# ── Copy models to local scratch ───────────────────────────────────
sh $SCRIPT_DIR/copy_models.sh 

# ── Start Ollama server ────────────────────────────────────────────
$OLLAMA_BIN serve &
OLLAMA_PID=$!
echo "Started ollama serve (PID $OLLAMA_PID)"
sleep 8

# ── Run inference demos ───────────────────────────────────────────
# Models are loaded from $OLLAMA_MODELS (set in config.sh).
# Run 06_ollama/pull_models.sh once beforehand if not yet downloaded.
$MICROMAMBA_BIN run -p $ENV_PREFIX \
    python $SCRIPT_DIR/inference.py   \
        --host     http://$OLLAMA_HOST \
        --data-dir $DATA_DIR

# ── Clean up ──────────────────────────────────────────────────────
echo ""
echo "Stopping ollama server (PID $OLLAMA_PID)..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null
echo "Done."
