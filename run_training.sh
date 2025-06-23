#!/bin/bash
#SBATCH --account=ctb-simine
#SBATCH --mem-per-gpu=40GB
<<<<<<< HEAD
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00
#SBATCH --array=0-4
#SBATCH --output=slurm_tr-%a.out
#SBATCH --error=slurm_tr-%a.err
=======
#SBATCH --array=0-4
#SBATCH --job-name=vanilla
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00
#SBATCH --output=slurm_tr.out
#SBATCH --error=slurm_tr.err
>>>>>>> vanilla_first

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

<<<<<<< HEAD
hp_values=(40 50 60 70 80)
hp_select=${hp_values[$SLURM_ARRAY_TASK_ID]}


python ./main.py --run_num=$SLURM_ARRAY_TASK_ID --experiment_name="rot_90_conv_layers_${hp_select}" --conv_layers="${hp_select}" --max_epochs=3000 --nrot=4
=======
nn=(1 10 20 40 59)
nvanilla="${nn[$SLURM_ARRAY_TASK_ID]}"
nequiv=$((60 - $nvanilla))

python ./main.py --run_num=0 --experiment_name="rot_90_conv_layers_60_nvanilla_${nvanilla}" --max_epochs=3000 --nrot=4 --equivariant_layers="$nequiv" --vanilla_layers="$nvanilla"
>>>>>>> vanilla_first
