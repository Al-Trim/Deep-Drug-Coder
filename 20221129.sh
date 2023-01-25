#!/bin/bash

#SBATCH --job-name=1129_job

export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/GASA
export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/ddc_pub
module load miniconda/qyy_ddc

# 获取retrain次数
python cRNN_train_20221129.py 'load_data' 0
retrain_times=`cat retrain_times`
rm retrain_times

sbatch 20221129_retrain.sh $retrain_times

