#!/bin/bash -l
#SBATCH --job-name=eval_bm25
#SBATCH --account=project_465002928
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_bm25_%j.log
#SBATCH --error=logs/eval_bm25_%j.err

# BM25 text retrieval evaluation (CPU-only, no GPU needed).
# Run: sbatch slurm/eval_bm25.sh

set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-/scratch/project_465002928/$(whoami)/danlonben}"
VENV="${VENV:-$HOME/venvs/contextual-rag}"

mkdir -p "$PROJ_ROOT/logs"

source "$VENV/bin/activate"
export DANLONBEN_PROJ_ROOT="$PROJ_ROOT"
cd "$PROJ_ROOT"

python -u -m danlonben.retrieval.run_eval \
    --retriever bm25
