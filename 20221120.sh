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

#SBATCH --job-name=1120_job

export PYTHONPATH=${PYTHONPATH}:~/ddc/GASA
export PYTHONPATH=${PYTHONPATH}:~/ddc/ddc_pub
source ~/.bashrc
source activate ~/Miniconda3/envs/ddc_env

# 获取retrain次数
python cRNN_train_20221120.py 'load_data' 0
retrain_times=`cat retrain_times`
rm retrain_times

sbatch 20221120_retrain.sh $retrain_times

