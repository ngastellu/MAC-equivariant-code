#!/bin/bash
#SBATCH --account=ctb-simine

#SBATCH --mem-per-gpu=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-72:00
#SBATCH --array=0

module load httpproxy
module load cuda cudnn
source /home/ataminer/env4/bin/activate


module load httpproxy

export MASTER_PORT=15109
# WORLD_SIZE as gpus/node * num_nodes
export WORLD_SIZE=1

### get the first node name as master address - customized for vgg slurm #SBATCH --cpus-per-task=1
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

python ./main.py --run_num=$SLURM_ARRAY_TASK_ID 



