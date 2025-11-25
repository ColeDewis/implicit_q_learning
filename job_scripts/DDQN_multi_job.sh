#!/bin/bash

# Example usage:
# sbatch --export=path="$(pwd)" DDQN_multi_job.sh 2 antmaze-large-play-v0 Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5 01:10:00

# Set the number of seeds dynamically (first argument)
NUM_SEEDS=${1:-3}  # Default to 3 seeds if not provided

# Set the environment name (second argument)
ENV_NAME=${2:-antmaze-large-play-v0}  # Default to "antmaze-large-play-v0" if not provided

# Set the dataset name (third argument)
DATASET_NAME=${3:-Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5}  # Default dataset

# Set the job time dynamically (fourth argument)
JOB_TIME=${4:-01:00:00}  # Default to "01:00:00" if not provided

# Hyperparameters need to match what is in the configs files.
CONFIGS=(
    "CEM_AM_10_10_5"
    "CEM_AM_10_20_10"
    "CEM_AM_10_30_15"
    "AC_AM"
    "GA_AM"
)

#SBATCH --array=0-$((${#CONFIGS[@]}-1))
#SBATCH --time=${JOB_TIME}
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=l40s
#SBATCH --mem=16G
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

tar -xvf venv310.tar
source .venv/bin/activate

RESULTS_DIR=$path/results/hyper_sweep/${ENV_NAME}_${DATASET_NAME%.*}/
mkdir -p $RESULTS_DIR

# Get the hyperparameter combination for this job
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Training loop for multiple seeds per hyperparameter
for ((i=0; i<NUM_SEEDS; i++)); do
    SEED=$i  # Start seeds at 0
    python $path/train_offline.py --env_name=$ENV_NAME --config=$path/configs/$CONFIG.py --max_steps=100 --eval_episodes=100 --eval_interval=100000 --seed=$SEED --learner=DDQN
    RESULT_FILE=$RESULTS_DIR/seed${SEED}-env=${ENV_NAME}-hypers=${CONFIG}.txt
    cp ./tmp/${SEED}_${SLURM_ARRAY_TASK_ID}.txt $RESULT_FILE
done