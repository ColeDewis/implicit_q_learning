#!/bin/bash

#SBATCH --array=1-10:2
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2 

module load python/3.10
module load mujoco/3.1.6
module load cuda
export MUJOCO_PATH=~/.mujoco/mjpro150
export MUJOCO_PLUGIN_PATH=~/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin/:$LD_LIBRARY_PATH
export D4RL_DATASET_DIR=$SLURM_TMPDIR

cp $path/venv310.tar $SLURM_TMPDIR/
cp ~/.d4rl/datasets/halfcheetah_medium_expert-v2.hdf5 $SLURM_TMPDIR/
cd $SLURM_TMPDIR

tar -xvf venv310.tar
source .venv/bin/activate

mkdir -p $path/results/IQL/halfcheetah_medium_expert/

# NOTE: CC seems to want jobs at least ~1hr.
python $path/train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=$path/configs/mujoco_config.py --seed=$SLURM_ARRAY_TASK_ID
cp ./tmp/${SLURM_ARRAY_TASK_ID}.txt $path/results/IQL/halfcheetah_medium_expert/

SECOND_SEED=$((SLURM_ARRAY_TASK_ID + 1))
python $path/train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=$path/configs/mujoco_config.py --seed=$SECOND_SEED
cp ./tmp/${SECOND_SEED}.txt $path/results/IQL/halfcheetah_medium_expert/