#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --job-name=1129_retrain

#export PYTHONPATH=${PYTHONPATH}:~/ddc/GASA
export PYTHONPATH=${PYTHONPATH}:~QinYY/ddc/GASA
#export PYTHONPATH=${PYTHONPATH}:~/ddc/ddc_pub
export PYTHONPATH=${PYTHONPATH}:~QinYY/ddc/ddc_pub
module load miniconda/qyy_ddc

# 获取retrain次数
retrain_times=$1

for i in `seq 0 $retrain_times`
do
    python cRNN_train_20221129.py 'retrain' $i
    
    # 获取生成的分子数以便切割
    n=`cat generated_mols_$i`
    rm generated_mols_$i
    sbatch 20221129_calc.sh $i $n
done



