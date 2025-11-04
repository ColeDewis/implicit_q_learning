#!/bin/bash

# I ran this on fir and nibi.

#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3 

module load python/3.10
module load mujoco/3.1.6
module load cuda
export MUJOCO_PATH=~/.mujoco/mjpro150
export MUJOCO_PLUGIN_PATH=~/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin/:$LD_LIBRARY_PATH
export D4RL_DATASET_DIR=$SLURM_TMPDIR

cp $path/venv310.tar $SLURM_TMPDIR/
cp ~/.d4rl/datasets/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5 $SLURM_TMPDIR/
cd $SLURM_TMPDIR

tar -xvf venv310.tar
source .venv/bin/activate

python $path/train_offline.py --env_name=antmaze-large-play-v0 --config=$path/configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000

cp ./tmp/42.txt $path/