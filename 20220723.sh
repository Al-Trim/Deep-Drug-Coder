#!/bin/bash
##SBATCH --job-name=hello_world
#SBATCH --output=20220723.out
#SBATCH --error=20220723.err
##SBATCH --mail-type=ALL 
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=4
##SBATCH --ntasks-per-node=1
#SBATCH --distribution=cyclic:cyclic
##SBATCH --mem-per-cpu=500mb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:2

module load miniconda
source activate ../ddc_env_qyy

python cRNN_train_20220723.py
