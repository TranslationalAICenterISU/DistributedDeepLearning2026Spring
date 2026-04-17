#!/bin/bash
# One-time model download script — run this from a login node BEFORE the workshop.
# Models are stored in $OLLAMA_MODELS (set in config.sh).
# Subsequent jobs load from that path; no internet access needed on compute nodes.
#
# Usage:
#   bash 06_ollama/pull_models.sh

set -e
source config.sh

echo "Ollama bin   : $OLLAMA_BIN"
echo "Model store  : $OLLAMA_MODELS"
echo ""

mkdir -p "$OLLAMA_MODELS"

# Start a temporary server pointing at the model store
$OLLAMA_BIN serve &
OLLAMA_PID=$!
echo "Started ollama serve (PID $OLLAMA_PID)"
sleep 8

for MODEL in gpt-oss:20b qwen3.6 gemma4; do
    echo ""
    echo "──────────────────────────────────────"
    echo "Pulling $MODEL ..."
    $OLLAMA_BIN pull $MODEL
    echo "Done: $MODEL"
done

echo ""
echo "──────────────────────────────────────"
echo "All models stored in: $OLLAMA_MODELS"
du -sh "$OLLAMA_MODELS"

kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null
echo "Server stopped."
