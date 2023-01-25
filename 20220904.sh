#!/bin/bash
##SBATCH --mail-type=ALL 
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=4
##SBATCH --ntasks-per-node=1
#SBATCH --distribution=cyclic:cyclic
##SBATCH --mem-per-cpu=500mb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

#SBATCH --job-name=cRNN_train_20220904
#SBATCH --output=20220904.out
#SBATCH --error=20220904.err

module load miniconda
source activate ~/QinYY/ddc_env_qyy

python cRNN_train_20220904.py
