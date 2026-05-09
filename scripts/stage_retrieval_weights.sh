#!/bin/bash
# Download retrieval model weights to HF_SCRATCH on the login node.
# Compute nodes have no internet — run this ONCE before submitting eval jobs.
#
# Run: bash scripts/stage_retrieval_weights.sh

set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif
VENV="${VENV:-$HOME/venvs/contextual-rag}"
HF_SCRATCH="${HF_SCRATCH:-/scratch/project_465002928/$(whoami)/hf}"

mkdir -p "$HF_SCRATCH"

echo ">>> Staging retrieval model weights to $HF_SCRATCH"

singularity exec "$SIF" bash -c "
    source '$VENV/bin/activate'
    export HF_HOME='$HF_SCRATCH'
    export TRANSFORMERS_CACHE='$HF_SCRATCH'

    echo '>>> Downloading BAAI/bge-m3 ...'
    python -c '
from sentence_transformers import SentenceTransformer
SentenceTransformer(\"BAAI/bge-m3\")
print(\"bge-m3: done\")
'

    echo '>>> Downloading vidore/colpali-v1.2 ...'
    python -c '
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
ColPaliProcessor.from_pretrained(\"vidore/colpali-v1.2\")
ColPali.from_pretrained(\"vidore/colpali-v1.2\", torch_dtype=torch.bfloat16)
print(\"colpali-v1.2: done\")
'

    echo '>>> Downloading vidore/colqwen2-v1.0 ...'
    python -c '
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
ColQwen2Processor.from_pretrained(\"vidore/colqwen2-v1.0\")
ColQwen2.from_pretrained(\"vidore/colqwen2-v1.0\", torch_dtype=torch.bfloat16)
print(\"colqwen2-v1.0: done\")
'

    echo '>>> All retrieval weights staged.'
"
