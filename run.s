#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=CVAss2
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#####SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out.CVAss2.%j

module load pytorch/python2.7/0.3.0_4
module load python/intel/2.7.12 pytorch/0.2.0_1 protobuf/intel/3.1.0 spyder/3.1.4
module load torchvision/0.1.8

python main.py
python evaluate.py --model model_latest_Adagrad_dataAugmentation_lr.pth --outfile out_latest_Adagrad_dataAugmentation_lr.csv
