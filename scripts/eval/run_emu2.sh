#!/bin/bash -l
set -e
# echo "Sourcing conda..."
source ~/miniconda3/etc/profile.d/conda.sh


# Emu2
conda deactivate
cd /home/colligo/SpotFrame/models/Emu2
conda activate emu2
export CUDA_VISIBLE_DEVICES=4
python edit_inference.py