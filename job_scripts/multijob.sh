#!/bin/bash

# Example usage:
# sbatch --export=path="$(pwd)" multijob.sh 3 antmaze-large-play-v0 Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5 01:40:00

# Set the step size dynamically (first argument)
STEP_SIZE=${1:-3}  # Default to 3 if not provided

# Set the environment name (second argument)
ENV_NAME=${2:-antmaze-large-play-v0}  # Default to "antmaze-large-play-v0" if not provided

# Set the dataset name (third argument)
DATASET_NAME=${3:-Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5}  # Default dataset

# Set the job time dynamically (fourth argument)
JOB_TIME=${4:-01:40:00}  # Default to "01:40:00" if not provided

#SBATCH --array=1-15:${STEP_SIZE}
#SBATCH --time=${JOB_TIME}
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=3

# Load required modules
module load python/3.10
module load mujoco/3.1.6
module load cuda

# Environment variables for MuJoCo and dataset
export MUJOCO_PATH=~/.mujoco/mjpro150
export MUJOCO_PLUGIN_PATH=~/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin/:$LD_LIBRARY_PATH
export D4RL_DATASET_DIR=$SLURM_TMPDIR

# Copy virtual environment and dataset to temporary directory
cp $path/venv310.tar $SLURM_TMPDIR/
cp ~/.d4rl/datasets/$DATASET_NAME $SLURM_TMPDIR/
cd $SLURM_TMPDIR

# Extract virtual environment and activate it
tar -xvf venv310.tar
source .venv/bin/activate

# Create results directory
RESULTS_DIR=$path/results/IQL/${ENV_NAME}_${DATASET_NAME%.*}/
mkdir -p $RESULTS_DIR

# Training loop for multiple seeds
for ((i=0; i<STEP_SIZE; i++)); do
    SEED=$((SLURM_ARRAY_TASK_ID + i))
    python $path/train_offline.py --env_name=$ENV_NAME --config=$path/configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000 --seed=$SEED
    cp ./tmp/${SEED}.txt $RESULTS_DIR
done