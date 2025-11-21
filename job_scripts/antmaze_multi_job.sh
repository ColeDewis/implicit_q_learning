#!/bin/bash

#SBATCH --array=1-15:3
#SBATCH --time=01:40:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --mem=8G
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

mkdir -p $path/results/IQL/Ant_maze_hardest_noisy_multistart/

# NOTE: CC seems to want jobs at least ~1hr.
python $path/train_offline.py --env_name=antmaze-large-play-v0 --config=$path/configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000 --seed=$SLURM_ARRAY_TASK_ID
cp ./tmp/${SLURM_ARRAY_TASK_ID}.txt $path/results/IQL/Ant_maze_hardest_noisy_multistart/

SECOND_SEED=$((SLURM_ARRAY_TASK_ID + 1))
python $path/train_offline.py --env_name=antmaze-large-play-v0 --config=$path/configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000 --seed=$SECOND_SEED
cp ./tmp/${SECOND_SEED}.txt $path/results/IQL/Ant_maze_hardest_noisy_multistart/

THIRD_SEED=$((SLURM_ARRAY_TASK_ID + 2))
python $path/train_offline.py --env_name=antmaze-large-play-v0 --config=$path/configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000 --seed=$THIRD_SEED
cp ./tmp/${THIRD_SEED}.txt $path/results/IQL/Ant_maze_hardest_noisy_multistart/