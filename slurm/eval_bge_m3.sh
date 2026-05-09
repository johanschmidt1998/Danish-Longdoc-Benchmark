#!/bin/bash -l
#SBATCH --job-name=eval_bge_m3
#SBATCH --account=project_465002928
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_bge_m3_%j.log
#SBATCH --error=logs/eval_bge_m3_%j.err

# BGE-M3 dense text retrieval evaluation.
# Pre-stage model weights on the login node first:
#   bash scripts/stage_retrieval_weights.sh
#
# Run: sbatch slurm/eval_bge_m3.sh

set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

PROJ_ROOT="${PROJ_ROOT:-/scratch/project_465002928/$(whoami)/danlonben}"
HF_SCRATCH="${HF_SCRATCH:-/scratch/project_465002928/$(whoami)/hf}"
VENV="${VENV:-$HOME/venvs/contextual-rag}"

mkdir -p "$PROJ_ROOT/logs"

srun singularity exec "$SIF" bash -c "
    if [ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ] && [ -z \"\${HIP_VISIBLE_DEVICES:-}\" ]; then
        export HIP_VISIBLE_DEVICES=\"\$ROCR_VISIBLE_DEVICES\"
    fi
    unset ROCR_VISIBLE_DEVICES
    source '$VENV/bin/activate'
    export HF_HOME='$HF_SCRATCH'
    export TRANSFORMERS_CACHE='$HF_SCRATCH'
    export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
    export DANLONBEN_PROJ_ROOT='$PROJ_ROOT' && cd '$PROJ_ROOT' && python -u -m danlonben.retrieval.run_eval \
        --retriever bge-m3 \
        --device cuda \
        --batch-size 32
"
