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

#SBATCH --job-name=cRNN_test_20221115

export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/GASA
export PYTHONPATH=${PYTHONPATH}:~/QinYY/ddc/ddc_pub
module load miniconda/qyy_ddc

for i in `seq 0 3`
do
    python cRNN_train_20221115.py 'generate' $i
    ###获取生成的分子数以便切割
    n=`cat generated_mols_$i`
    rm generated_mols_$i
    sbatch 20221115_submit.sh $i $n
done



