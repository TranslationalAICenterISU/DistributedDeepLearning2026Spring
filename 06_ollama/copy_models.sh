#!/bin/bash
# One-time model copy script — run this at the start of a job
# Models are stored in $OLLAMA_MODELS (set in config.sh).
# Subsequent jobs load from that path; no internet access needed on compute nodes.
#
# Usage:
#   bash 06_ollama/pull_models.sh

set -e
source ../config.sh

echo "Ollama bin   : $OLLAMA_BIN"
echo "Model store  : $OLLAMA_MODELS"
echo "Model source : $OLLAMA_MODELS_STAGE"
echo ""

mkdir -p "$OLLAMA_MODELS"
echo "──────────────────────────────────────"
echo "Copying models to: $OLLAMA_MODELS"
rsync -a $OLLAMA_MODELS_STAGE/* $OLLAMA_MODELS/

# Start a temporary server pointing at the model store
$OLLAMA_BIN serve &
OLLAMA_PID=$!
echo "Started temporary ollama server (PID $OLLAMA_PID)"
echo ""
echo "──────────────────────────────────────"
echo "All models stored in: $OLLAMA_MODELS"
du -sh "$OLLAMA_MODELS"
echo ""
echo "──────────────────────────────────────"
echo "Listing models"
$OLLAMA_BIN list 

# Cleanup
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null
echo "Temporary server stopped."
