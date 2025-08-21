#!/bin/bash -l

cd models/BAGEL
python edit_inference.py --mode {real or syn}

cd models/Emu2  
python edit_inference.py --mode {real or syn}

cd models/OmniGen 
python edit_inference.py --mode {real or syn}

cd models/OmniGen2
python edit_inference.py --mode {real or syn}

export HF_HOME=<cache>
export HF_TOKEN=<token>
cd models/UNO
python edit_inference.py --mode real
