# ── Workshop path configuration ────────────────────────────────────
# This file is sourced by every sbatch script.
# Only this file needs to be edited when changing environments.

# Base working directory.
#   Instructor  : /work/mech-ai/Aditya
#   Participants: /work/short_term
GROUP_DIR=/work/mech-ai/Aditya

# micromamba binary (load via 'module load micromamba' for participants)
MICROMAMBA_BIN=/opt/rit/el9/20240413/app/linux-rhel9-x86_64_v3/gcc-11.3.1/micromamba-1.4.2-lcemqbetrry5tmxx3oygtoc3apv2af7e/bin/micromamba

# Python environment prefix.
#   Instructor  : ${GROUP_DIR}/Micromamba_demo
#   Participants: ${GROUP_DIR}/$USER/mamba_envs/dl_workshop
ENV_PREFIX=${GROUP_DIR}/Micromamba_demo

# Shared data and checkpoint directories
DATA_DIR=${GROUP_DIR}/data
SAVE_BASE=${GROUP_DIR}/checkpoints

# Ollama binary and model storage
OLLAMA_BIN=/home/baditya/2025_DDP/inference/bin/ollama
OLLAMA_MODELS=${GROUP_DIR}/ollama/models
export OLLAMA_MODELS
export LD_LIBRARY_PATH=/home/baditya/2025_DDP/inference/lib/ollama:$LD_LIBRARY_PATH
