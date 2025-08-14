#!/bin/bash -l

# cd models/BAGEL
#/scratch/sg7457/code/SpotEdit/pytorch-example/python edit_inference.py --mode $1

# cd models/Emu2  
# /scratch/sg7457/code/SpotEdit/pytorch-example/python-emu2 edit_inference.py --mode $1

# cd models/OmniGen 
# /scratch/sg7457/code/SpotEdit/pytorch-example/python-omnigen edit_inference.py --mode $1

# cd models/OmniGen2  
# /scratch/sg7457/code/SpotEdit/pytorch-example/python-omnigen2 edit_inference.py --mode $1
export HF_HOME=/scratch/sg7457/code/SpotEdit/saved_models/UNO
export HF_TOKEN=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
cd models/UNO
/scratch/sg7457/code/SpotEdit/pytorch-example/python edit_inference.py --mode $1
