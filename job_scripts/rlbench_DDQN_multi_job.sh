#!/bin/bash

#SBATCH --array=1-15:3
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=3

# Example usage:
# sbatch --export=path="$(pwd)" rlbench_DDQN_multi_job.sh 3 microwave microwave_data.h5 03:00:00

# Set the number of seeds dynamically (first argument)
NUM_SEEDS=${1:-3}  # Default to 3 seeds if not provided

# Set the environment name (second argument)
ENV_NAME=${2:-microwave}  # Default to "antmaze-large-play-v0" if not provided

# Set the dataset name (third argument)
DATASET_NAME=${3:-microwave_data.h5}  # Default dataset

# Set the job time dynamically (fourth argument)
JOB_TIME=${4:-02:00:00}  # Default to "01:00:00" if not provided

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
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=3

# Load required modules
module load python/3.10
module load mujoco/3.1.6
module load cuda

# Environment variables for MuJoCo and dataset
setup_iql_cmds="export MUJOCO_PATH=~/.mujoco/mjpro150; \
export MUJOCO_PLUGIN_PATH=~/.mujoco/mjpro150/bin; \
export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin/:\$LD_LIBRARY_PATH; \
export D4RL_DATASET_DIR=$SLURM_TMPDIR \
export MUJOCO_GL=egl \
export PYOPENGL_PLATFORM=egl"

setup_rlbench_cmds="export COPPELIASIM_ROOT=${HOME}/CoppeliaSim \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT \
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT"

# Copy virtual environment and dataset to temporary directory
cp $path/venv310.tar $SLURM_TMPDIR/
cp $path/../RLBench/venv_rlbench.tar $SLURM_TMPDIR/
cp ~/.d4rl/datasets/$DATASET_NAME $SLURM_TMPDIR/
cd $SLURM_TMPDIR

# Extract virtual environment and activate it
tar -xf venv310.tar
tar -xf venv_rlbench.tar

# Create results directory
RESULTS_DIR=$path/results/hyper_sweep/${ENV_NAME}_${DATASET_NAME%.*}/
mkdir -p $RESULTS_DIR

# Get the hyperparameter combination for this job
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Training loop for multiple seeds per hyperparameter
for ((i=0; i<NUM_SEEDS; i++)); do
    SEED=$i  # Start seeds at 0
    PORT=$((i+5000))
    RESULT_FILE=$RESULTS_DIR/seed${SEED}-env=${ENV_NAME}-hypers=${CONFIG}.txt

    # tmux will have 2 instances per session
    # Instance 1 will have iql repo:
    tmux new-session -d -s $SESSION_NAME
    tmux send-keys -t ${SESSION_NAME}:0.0 "cd $SLURM_TMPDIR" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "source .venv/bin/activate" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "$setup_iql_cmds" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "python $path/train_offline.py --env_name=$ENV_NAME --config=$path/configs/$CONFIG.py --eval_episodes=100 --eval_interval=100000  --seed=$SEED --port=$PORT --learner=DDQN" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "cp ./tmp/${SEED}_${SLURM_ARRAY_TASK_ID}.txt $RESULT_FILE" C-m


    # Instance 2 will have rlbench repo:
    tmux split-window -h -t ${SESSION_NAME}:0
    tmux send-keys -t ${SESSION_NAME}:0.1 "cd $SLURM_TMPDIR" C-m
    tmux send-keys -t ${SESSION_NAME}:0.1 "source .venv_rlbench/bin/activate" C-m
    tmux send-keys -t ${SESSION_NAME}:0.1 "$setup_rlbench_cmds" C-m
    tmux send-keys -t ${SESSION_NAME}:0.1 "xvfb-run -a python $path/../RLBench/env_server.py --port=$PORT" C-m
done

echo "Waiting for Tmux sessions to complete..."
while [ $(tmux list-sessions 2>/dev/null | grep pair_seed | wc -l) -gt 0 ]; do
    sleep 10
done

echo "All sessions completed."
