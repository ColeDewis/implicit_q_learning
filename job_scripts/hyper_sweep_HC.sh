#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus-per-node=l40s
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --mail-user=jleng1@ualberta.ca
#SBATCH --mail-type=END,FAIL

# Example usage:
# sbatch --time=01:00:00 --array=1-28 --export=path="$(pwd)" job_scripts/hyper_sweep.sh 2 antmaze-large-play-v0 Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5 antmaze_config.py

# Set the number of seeds dynamically (first argument)
NUM_SEEDS=${1:-2}  # Default to 2 seeds if not provided

# Set the environment name (second argument)
ENV_NAME=${2:-antmaze-large-play-v0}  # Default to "antmaze-large-play-v0" if not provided

# Set the dataset name (third argument)
DATASET_NAME=${3:-Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5}  # Default datase

# Set the config file to be used
CONFIG=${4:-CEM_AM_10_20_10} # Default CEM on AntMaze with 10 iterations, 10 samples, and 5 elite


# Hyperparameters need to match what is in the configs files.
HYPERPARAMS=(
    ""
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=1,tau=0.0025"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=1,tau=0.005"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=1,tau=0.0075"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=3,tau=0.0025"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=3,tau=0.005"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=3,tau=0.0075"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=5,tau=0.0025"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=5,tau=0.005"
    "actor_lr=0.001,critic_lr=0.001,value_lr=0.001,temperature=5,tau=0.0075"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=1,tau=0.0025"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=1,tau=0.005"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=1,tau=0.0075"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=3,tau=0.0025"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=3,tau=0.005"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=3,tau=0.0075"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=5,tau=0.0025"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=5,tau=0.005"
    "actor_lr=0.0003,critic_lr=0.0003,value_lr=0.0003,temperature=5,tau=0.0075"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=1,tau=0.0025"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=1,tau=0.005"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=1,tau=0.0075"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=3,tau=0.0025"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=3,tau=0.005"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=3,tau=0.0075"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=5,tau=0.0025"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=5,tau=0.005"
    "actor_lr=0.0001,critic_lr=0.0001,value_lr=0.0001,temperature=5,tau=0.0075"

)

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
HYPERPARAM=${HYPERPARAMS[$SLURM_ARRAY_TASK_ID]}

# Format hyperparameters for file naming (replace commas with underscores)
HYPERPARAM_FORMATTED=$(echo $HYPERPARAM | tr ',' '-')

# Training loop for multiple seeds per hyperparameter
for ((i=0; i<NUM_SEEDS; i++)); do
    SEED=$i  # Start seeds at 0
    python $path/train_offline.py --env_name=$ENV_NAME --config=$path/configs/$CONFIG.py --learner=DDQN --eval_episodes=100 --eval_interval=1000000 --seed=$SEED --overrides=$HYPERPARAM
    RESULT_FILE=$RESULTS_DIR/seed${SEED}-env=${ENV_NAME}-hypers=${HYPERPARAM_FORMATTED}.txt
    cp ./tmp/${SEED}_${SLURM_ARRAY_TASK_ID}.txt $RESULT_FILE
done