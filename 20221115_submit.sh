#!/bin/bash
#SBATCH --job-name=1115_submit

export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/GASA
export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/ddc_pub
module load parallel
module load miniconda/qyy_ddc

i=$1
n=$2
n=`expr $n / 100`
echo $n

for idx in `seq 0 $n`
do 
    echo python cRNN_train_20221115.py 'generate_calc' $i $idx
done | parallel

python cRNN_train_20221115.py 'generate_report' $i $n
