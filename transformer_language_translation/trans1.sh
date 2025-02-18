#!/bin/bash
#SBATCH -A shaon
#SBATCH --nodelist=gnode057
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=96:00:00
#SBATCH --output trans1.txt


source activate trans
which python



python train.py