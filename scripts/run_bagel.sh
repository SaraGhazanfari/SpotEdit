#!/bin/bash -l
set -e

source ~/miniconda3/etc/profile.d/conda.sh

# BAGEL
conda deactivate
cd /home/colligo/SpotFrame/models/BAGEL
conda activate bagel
export CUDA_VISIBLE_DEVICES=$1
python edit_inference.py