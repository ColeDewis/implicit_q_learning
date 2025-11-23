#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=3

# Example usage:
# sbatch --time=01:00:00 --array=1-10:2 --export=path="$(pwd)" job_scripts/multijob.sh 2 antmaze-large-play-v0 Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5 antmaze_config.py 100000
# Note that you MUST pass consistent datasets and environment names, and the number
# (step size) after the script name must match the array step size.
# Run this from the root repository folder.

# Set the step size dynamically (first argument)
STEP_SIZE=${1:-2}  # Default to 2 if not provided

# Set the environment name (second argument)
ENV_NAME=${2:-antmaze-large-play-v0}  # Default to "antmaze-large-play-v0" if not provided

# Set the dataset name (third argument)
DATASET_NAME=${3:-Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5}  # Default dataset

# Set the config name dynamically (fourth argument)
CONFIG_NAME=${4:-antmaze_finetune_config.py}  # Default to "antmaze_finetune_config.py" if not provided

# Set the evaluation interval dynamically (fifth argument)
EVAL_INTERVAL=${5:-100000}  # Default to 100000 if not provided

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
    python $path/train_finetune.py --env_name=$ENV_NAME --config=$path/configs/${CONFIG_NAME} --eval_episodes=100 --replay_buffer_size 2000000 --eval_interval=${EVAL_INTERVAL} --seed=$SEED

    cp ./tmp/IQL_${SEED}.txt $RESULTS_DIR
done