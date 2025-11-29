#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=3
#SBATCH --mail-user=khlynovs@ualberta.ca
#SBATCH --mail-type=END,FAIL

# Example usage:
# sbatch --time=07:00:00 --array=1-28 --export=path="$(pwd)" job_scripts/rlbench_hyper_sweep.sh 2 microwave microwave_data.h5 CEM_AM_10_20_10
# Set the number of seeds dynamically (first argument)
STEP_SIZE=${1:-2}  # Default to 2 if not provided

# Set the environment name (second argument)
ENV_NAME=${2:-microwave}  # Default to "antmaze-large-play-v0" if not provided

# Set the dataset name (third argument)
DATASET_NAME=${3:-microwave_data.h5}  # Default datase

# Set the config file to be used
CONFIG=${4:-CEM_AM_10_20_10} # Default CEM on AntMaze with 10 iterations, 10 samples, and 5 elite


# Hyperparameters need to match what is in the configs files.
HYPERPARAMS=(
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=5,tau=0.0025"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=5,tau=0.005"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=5,tau=0.0075"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=10,tau=0.0025"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=10,tau=0.005"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=10,tau=0.0075"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=15,tau=0.0025"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=15,tau=0.005"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=15,tau=0.0075"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=5,tau=0.0025"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=5,tau=0.005"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=5,tau=0.0075"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=10,tau=0.0025"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=10,tau=0.005"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=10,tau=0.0075"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=15,tau=0.0025"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=15,tau=0.005"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=15,tau=0.0075"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=5,tau=0.0025"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=5,tau=0.005"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=5,tau=0.0075"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=10,tau=0.0025"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=10,tau=0.005"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=10,tau=0.0075"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=15,tau=0.0025"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=15,tau=0.005"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=15,tau=0.0075"

)

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
mkdir -p tmp

# Get the hyperparameter combination for this job
HYPERPARAM=${HYPERPARAMS[$SLURM_ARRAY_TASK_ID]}

# Format hyperparameters for file naming (replace commas with underscores)
HYPERPARAM_FORMATTED=$(echo $HYPERPARAM | tr ',' '-')

# Training loop for multiple seeds
for ((i=0; i<STEP_SIZE; i++)); do
    SEED=$((SLURM_ARRAY_TASK_ID + i))
    PORT=$((i+SLURM_ARRAY_TASK_ID+5000))
    SESSION_NAME="pair_seed_${SEED}"
    RESULT_FILE=$RESULTS_DIR/${CONFIG}seed${SEED}-env=${ENV_NAME}-hypers=${HYPERPARAM_FORMATTED}.txt
    echo "Starting Tmux session: $SESSION_NAME and saving $path/tmp/DDQN_${SEED}_${HYPERPARAM_FORMATTED}.txt in $RESULT_FILE"

    # tmux will have 2 instances per session
    # Instance 1 will have iql repo:
    tmux new-session -d -s $SESSION_NAME
    tmux send-keys -t ${SESSION_NAME}:0.0 "cd $SLURM_TMPDIR" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "source .venv/bin/activate" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "$setup_iql_cmds" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "python $path/train_offline.py --env_name=$ENV_NAME --config=$path/configs/$CONFIG.py --learner=DDQN --eval_episodes=100 --eval_interval=1000000  --seed=$SEED --port=$PORT --overrides=$HYPERPARAM" C-m
    tmux send-keys -t ${SESSION_NAME}:0.0 "cp ./tmp/DDQN_${SEED}_${HYPERPARAM_FORMATTED}.txt $RESULT_FILE" C-m


    # Instance 2 will have rlbench repo:
    tmux split-window -h -t ${SESSION_NAME}:0
    tmux send-keys -t ${SESSION_NAME}:0.1 "cd $SLURM_TMPDIR" C-m
    tmux send-keys -t ${SESSION_NAME}:0.1 "source .venv_rlbench/bin/activate" C-m
    tmux send-keys -t ${SESSION_NAME}:0.1 "$setup_rlbench_cmds" C-m
    tmux send-keys -t ${SESSION_NAME}:0.1 "xvfb-run -a python $path/../RLBench/env_server.py --port=$PORT --seed=$SEED" C-m
    # tmux send-keys -t ${SESSION_NAME}:0.1 "python $path/../RLBench/env_server.py --port=$PORT" C-m
done

echo "Waiting for Tmux sessions to complete..."
while [ $(tmux list-sessions 2>/dev/null | grep pair_seed | wc -l) -gt 0 ]; do
    sleep 10
done

echo "All sessions completed."
