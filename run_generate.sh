#!/bin/bash
#SBATCH --account=ctb-simine

#SBATCH --mem-per-gpu=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00
#SBATCH --array=0-1
#SBATCH --output=slrmgen-%a.out
#SBATCH --error=slrmgen-%a.err

module load httpproxy
module load cuda cudnn
module load httpproxy


virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt


export MASTER_PORT=19109
# WORLD_SIZE as gpus/node * num_nodes
export WORLD_SIZE=1

### get the first node name as master address - customized for vgg slurm #SBATCH --cpus-per-task=1
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

hp_values=(60 70)
hp_select=${hp_values[$SLURM_ARRAY_TASK_ID]}

python ./generate.py --run_num=$SLURM_ARRAY_TASK_ID --training_run_name="conv_layers_${hp_select}" --epoch_chkpt=-1 --n_samples=3 --softmax_temp=0.5
