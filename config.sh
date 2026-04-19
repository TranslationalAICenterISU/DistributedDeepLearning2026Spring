# ── Workshop path configuration ────────────────────────────────────
# This file is sourced by every sbatch script.
# Only this file needs to be edited when changing environments.

# Base working directory.
#   Instructor  : /work/mech-ai/Aditya
#   Participants: /work/short_term
GROUP_DIR=/work/short_term

# micromamba binary (load via 'module load micromamba' for participants)
MICROMAMBA_BIN=/opt/rit/el9/20240413/app/linux-rhel9-x86_64_v3/gcc-11.3.1/micromamba-1.4.2-lcemqbetrry5tmxx3oygtoc3apv2af7e/bin/micromamba

# Python environment prefix.
#   Instructor  : ${GROUP_DIR}/Micromamba_demo
#   Participants: ${GROUP_DIR}/$USER/mamba_envs/dl_workshop
ENV_PREFIX=${GROUP_DIR}/${USER}/mamba_envs/dl_workshop

# Shared data and checkpoint directories
DATA_DIR=${GROUP_DIR}/${USER}/data
SAVE_BASE=${GROUP_DIR}/${USER}/checkpoints

# Ollama binary and model storage
# Set OLLAMA_MODELS to the pre-downloaded model directory before running jobs.
# Run 06_ollama/pull_models.sh once (from a login node with internet) to populate it.
OLLAMA_BIN=${GROUP_DIR}/${USER}/ollama/bin/ollama

# Specify model-storage location
OLLAMA_MODELS_STAGE=/work/short_term/workshop_materials/deepl/ollama-models   # Pre-staged models downloaded by admins before the workshop
OLLAMA_MODELS=${TMPDIR}/ollama-models    # NOTE: This is job-ephemeral and unique per-job
#OLLAMA_MODELS=${GROUP_DIR}/${USER}/ollama/models   # ← update this path when finalised

# Modify to 127.X.Y.Z:AAAA (random values for X, Y, Z, A) to avoid clashing with other users on the same node
OLLAMA_HOST=127.2.2.2:58000   

# Export variables to the environment
export OLLAMA_MODELS
export OLLAMA_HOST
export LD_LIBRARY_PATH=/home/baditya/2025_DDP/inference/lib/ollama:$LD_LIBRARY_PATH
