#!/bin/bash -l
set -e
# echo "Sourcing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# OmniGen
conda deactivate
cd /home/colligo/SpotFrame/models/OmniGen
conda activate omnigen
export CUDA_VISIBLE_DEVICES=3
python edit_inference.py
