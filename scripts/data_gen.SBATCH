#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=sg7457@nyu.edu
#SBATCH --cpus-per-task=30
#SBATCH --time=48:00:00
#SBATCH --begin=now
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --job-name=validate
#SBATCH --mail-type=BEGIN
#SBATCH --account=pr_140_tandon_advanced


$SCRATCH/aa10460/pytorch-example/python -m data.main --step 5