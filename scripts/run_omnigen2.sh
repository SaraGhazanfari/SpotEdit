#!/bin/bash -l
set -e
# echo "Sourcing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# OmniGen2

conda deactivate
cd /home/colligo/SpotFrame/models/OmniGen2 
conda activate omnigen2
export CUDA_VISIBLE_DEVICES=0

model_path="OmniGen2/OmniGen2"
conda run -n omnigen2 
python edit_inference.py \
--model_path $model_path \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 5.0 \
--image_guidance_scale 2.0 \
--num_images_per_prompt 1 \
--start_idx 0
