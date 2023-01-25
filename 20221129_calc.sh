#!/bin/bash

#SBATCH --job-name=1129_calc

export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/GASA
export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/ddc_pub
module load miniconda/qyy_ddc
module load parallel/2019

i=$1
n=$2
n=`expr $n / 100`
echo "$i $n"

for idx in `seq 0 $n`
do 
    echo python cRNN_train_20221129.py 'retrain_calc' $i $idx
done | parallel

python cRNN_train_20221129.py 'retrain_report' $i $n
