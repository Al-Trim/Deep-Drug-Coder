#!/bin/bash

#SBATCH --job-name=train_main

export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/GASA
export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/ddc_pub
module load miniconda/qyy_ddc

model_name=$1
step='main'
# 获取retrain次数
python cRNN_train_$model_name.py $step
