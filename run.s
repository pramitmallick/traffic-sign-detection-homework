#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --verbose
#SBATCH --job-name=CVAss2
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
##SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out.CVAss2.%j
#SBATCH --error=err.log
 
module load intel/17.0.1
module load gcc/6.3.0
module load python3/intel/3.6.3
module load pytorch/python2.7/0.3.0_4
#module load pytorch/python3.6/0.3.0_4

python ./main.py