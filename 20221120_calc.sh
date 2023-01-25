#!/bin/bash

### 指定该作业需要多少个节点（申请多个节点的话需要您的程序支持分布式计算），必选项
#SBATCH --nodes=1

### 指定该作业在哪个分区队列上执行，gpu作业直接指定gpu分区即可，必选项
#SBATCH --partition=cpu

### 指定该作业运行多少个任务进程(默认为每个任务进程分配1个cpu核)，必选项
#SBATCH --ntasks=1

### 指定每个任务进程需要的cpu核数（默认1），可选项
#SBATCH --cpus-per-task=1

### 指定该作业从哪个项目扣费，如果没有这条参数，则从个人账户扣费
#SBATCH --comment=amber_project

#SBATCH --job-name=1120_calc

export PYTHONPATH=${PYTHONPATH}:~/ddc/GASA
export PYTHONPATH=${PYTHONPATH}:~/ddc/ddc_pub
source ~/.bashrc
source activate ~/Miniconda3/envs/ddc_env

i=$1
n=$2
n=`expr $n / 1000`
echo $n

for idx in `seq 0 $n`
do 
    echo python cRNN_train_20221120.py 'retrain_calc' $i $idx
done | parallel

python cRNN_train_20221120.py 'retrain_report' $i $n
