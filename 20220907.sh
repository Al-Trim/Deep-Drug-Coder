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

#SBATCH --job-name=cRNN_train_0907
#SBATCH --output=20220907.out
#SBATCH --error=20220907.err

module load app/cuda/10.1
##source /etc/bashrc
##source ~/.bashrc
##nvidia-smi
##export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/nvidia/lib

source activate ~/Miniconda3/envs/ddc_env
##source activate base
##python test.py
python cRNN_train_20220907.py
