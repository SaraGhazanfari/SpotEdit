#!/bin/bash -l

##DINO 
python -m metrics.spotedit_metric --mode syn --eval stan --encoder dino --pool cls
python -m metrics.spotedit_metric --mode syn --eval rob --encoder dino --pool cls

python -m metrics.spotedit_metric --mode real --eval stan --encoder dino --pool cls
python -m metrics.spotedit_metric --mode real --eval rob --encoder dino --pool cls


##CLIP 
python -m metrics.spotedit_metric --mode syn --eval stan --encoder clip --pool cls 
python -m metrics.spotedit_metric --mode real --eval rob --encoder clip --pool cls

python -m metrics.spotedit_metric --mode real --eval stan --encoder clip --pool cls
python -m metrics.spotedit_metric --mode syn --eval rob --encoder clip --pool cls







