#!/bin/bash -l
set -e
# echo "Sourcing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# UNO
conda deactivate
cd /home/colligo/SpotFrame/models/UNO
conda activate uno_env
export CUDA_VISIBLE_DEVICES=2
python edit_inference.py
