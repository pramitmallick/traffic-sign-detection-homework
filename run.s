#!/bin/bash
rm out.*
#SBATCH --verbose
#SBATCH --job-name=CVAss2
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
##SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out.CVAss2.%j

module load pytorch/python2.7/0.3.0_4

python main.py
python evaluate.py model_latest.pth
