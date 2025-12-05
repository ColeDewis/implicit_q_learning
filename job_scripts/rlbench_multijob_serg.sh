#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3

# Example usage:
# sbatch --time=06:00:00 --array=1-10:2 --export=path="$(pwd)" job_scripts/rlbench_multijob.sh 2 rlbench microwave_data.h5 mujoco_config.py 1000000
# Note that you MUST pass consistent datasets and environment names, and the number
# (step size) after the script name must match the array step size.
# Run this from the root repository folder.

# Set the step size dynamically (first argument)
STEP_SIZE=${1:-2}  # Default to 2 if not provided

# Set the environment name (second argument)
ENV_NAME=${2:-rlbench}  # Default to "antmaze-large-play-v0" if not provided

# Set the dataset name (third argument)
DATASET_NAME=${3:-microwave_data.h5}  # Default dataset

# Set the config name dynamically (fourth argument)
CONFIG_NAME=${4:-mujoco_config.py}  # Default to "antmaze_config.py" if not provided

# Set the evaluation interval dynamically (fifth argument)
EVAL_INTERVAL=${5:-100000}  # Default to 100000 if not provided

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

OVERRIDE="actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=5,tau=0.0075,expectile=0.9" # IQL
# OVERRIDE="" # CEM
# OVERRIDE="actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=5,tau=0.0025" # AM


# Extract virtual environment and activate it
tar -xf venv310.tar
tar -xf venv_rlbench.tar

# Create results directory
RESULTS_DIR=$path/results/IQL_finetune/${ENV_NAME}_${DATASET_NAME%.*}/
mkdir -p $RESULTS_DIR

LOGS_DIR=$path/logs/
mkdir -p $LOGS_DIR

# Training loop for multiple seeds
for ((i=0; i<STEP_SIZE; i++)); do
    SEED=$((SLURM_ARRAY_TASK_ID + i))
    PORT=$((i+SLURM_ARRAY_TASK_ID+5000))
    SESSION_NAME="pair_seed_${SEED}"
    HYPERPARAM_FORMATTED=$(echo $OVERRIDE | tr ',' '-')
    RESULT_FILE=$RESULTS_DIR/${CONFIG}seed${SEED}-env=${ENV_NAME}-hypers=${HYPERPARAM_FORMATTED}.txt
    LEARNER_LOG_FILE=$LOGS_DIR/learner_${SEED}.txt
    RLBENCH_LOG_FILE=$LOGS_DIR/rlbench_${SEED}.txt
    echo "Starting Tmux session: $SESSION_NAME"

    # tmux will have 2 instances per session
    # Instance 1 will have iql repo:
    tmux new-session -d -s $SESSION_NAME
    tmux send-keys -t ${SESSION_NAME}:0.0 "cd $SLURM_TMPDIR" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "source .venv/bin/activate" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "$setup_iql_cmds" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "python -u $path/train_finetune.py --env_name=$ENV_NAME --config=$path/configs/${CONFIG_NAME} --eval_episodes=100 --eval_interval=${EVAL_INTERVAL} --seed=$SEED --port=$PORT --overrides=$OVERRIDE 2>&1 | tee -a $LEARNER_LOG_FILE" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "cp ./tmp/IQL_${SEED}_${HYPERPARAM_FORMATTED}.txt $RESULT_FILE" C-m


    # Instance 2 will have rlbench repo:
    tmux split-window -h -t ${SESSION_NAME}:0
    tmux send-keys -t ${SESSION_NAME}:0.1 "cd $SLURM_TMPDIR" C-m
    tmux send-keys -t ${SESSION_NAME}:0.1 "source .venv_rlbench/bin/activate" C-m
    tmux send-keys -t ${SESSION_NAME}:0.1 "$setup_rlbench_cmds" C-m
    tmux send-keys -t ${SESSION_NAME}:0.1 "xvfb-run -a python -u $path/../RLBench/env_server.py --port=$PORT --seed=$SEED 2>&1 | tee -a $RLBENCH_LOG_FILE" C-m
done

echo "Waiting for Tmux sessions to complete..."
while [ $(tmux list-sessions 2>/dev/null | grep pair_seed | wc -l) -gt 0 ]; do
    sleep 10
done

echo "All sessions completed."
