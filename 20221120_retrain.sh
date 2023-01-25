#!/bin/bash

### 指定该作业需要多少个节点（申请多个节点的话需要您的程序支持分布式计算），必选项
#SBATCH --nodes=1

### 指定该作业在哪个分区队列上执行，gpu作业直接指定gpu分区即可，必选项
#SBATCH --partition=gpu

### 指定该作业运行多少个任务进程(默认为每个任务进程分配1个cpu核)，必选项
#SBATCH --ntasks=1

### 指定每个任务进程需要的cpu核数（默认1），可选项
#SBATCH --cpus-per-task=1

### 指定每个节点使用的GPU卡数量
### 喻园一号集群一个gpu节点最多可申请使用4张V100卡
### 强磁一号集群一个gpu节点最多可申请使用2张A100卡
### 数学交叉集群一个gpu节点最多可申请使用8张A100卡
#SBATCH --gres=gpu:1

### 指定该作业从哪个项目扣费，如果没有这条参数，则从个人账户扣费
#SBATCH --comment=amber_project

#SBATCH --job-name=1120_retrain

export PYTHONPATH=${PYTHONPATH}:~/ddc/GASA
export PYTHONPATH=${PYTHONPATH}:~/ddc/ddc_pub
source ~/.bashrc
source activate ~/Miniconda3/envs/ddc_env

# 获取retrain次数
retrain_times=$1

for i in `seq 0 $retrain_times`
do
    python cRNN_train_20221120.py 'retrain' $i
    
    # 获取生成的分子数以便切割
    n=`cat generated_mols_$i`
    rm generated_mols_$i
    sbatch 20221120_calc.sh $i $n
done



