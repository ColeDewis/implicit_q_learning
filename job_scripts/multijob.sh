#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=3

# Example usage:
# sbatch --time=01:00:00 --array=1-10:2 --export=path="$(pwd)" job_scripts/multijob.sh antmaze-large-play-v0 Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5 antmaze_config.py 100000

# Set the environment name (first argument)
ENV_NAME=${1:-antmaze-large-play-v0}  # Default to "antmaze-large-play-v0" if not provided

# Set the dataset name (second argument)
DATASET_NAME=${2:-Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5}  # Default dataset

# Set the config name (third argument)
CONFIG_NAME=${3:-antmaze_config.py}  # Default to "antmaze_config.py" if not provided

# Set the evaluation interval (fourth argument)
EVAL_INTERVAL=${4:-100000}  # Default to 100000 if not provided

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
    python $path/train_offline.py --env_name=$ENV_NAME --config=$path/configs/${CONFIG_NAME} --eval_episodes=100 --eval_interval=${EVAL_INTERVAL} --seed=$SEED
    cp ./tmp/${SEED}.txt $RESULTS_DIR
done