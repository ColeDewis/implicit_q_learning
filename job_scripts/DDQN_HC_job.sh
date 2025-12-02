#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus-per-node=l40s
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --mail-user=jleng1@ualberta.ca
#SBATCH --mail-type=END,FAIL

# Example usage:
# sbatch --time=07:00:00 --array=1-4  --export=path="$(pwd)" job_scripts/DDQN_HC_job.sh 11 TUNED_AC_HC

# Set the number of seeds dynamically (first argument)
NUM_SEEDS=${1:-11}  # Default to 3 seeds if not provided

# Set the config file to be used
CONFIG=${2:-TUNED_AC_HC} # Default CEM on AntMaze with 10 iterations, 10 samples, and 5 elite

ENVS = (
    "halfcheetah-medium-v2"
    "halfcheetah-medium-replay-v2"
    "halfcheetah-medium-expert-v2"
)

DATASETS = (
    "halfcheetah_medium-v2.hdf5"
    "halfcheetah_medium_replay-v2.hdf5"
    "halfcheetah_medium_expert-v2.hdf5"
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

ENV_NAME=${ENVS[$SLURM_ARRAY_TASK_ID]}
DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Copy virtual environment and dataset to temporary directory
cp $path/venv310.tar $SLURM_TMPDIR/
cp ~/.d4rl/datasets/$DATASET_NAME $SLURM_TMPDIR/
cd $SLURM_TMPDIR

tar -xvf venv310.tar
source .venv/bin/activate

RESULTS_DIR=$path/results/hyper_sweep/${ENV_NAME}_${DATASET_NAME%.*}/
mkdir -p $RESULTS_DIR

# Training loop for multiple seeds per hyperparameter
for ((i=1; i<NUM_SEEDS; i++)); do
    SEED=$i  # Start seeds at 0
    python $path/train_offline.py --env_name=$ENV_NAME --config=$path/configs/$CONFIG.py --eval_episodes=100 --eval_interval=100000 --seed=$SEED --learner=DDQN
    RESULT_FILE=$RESULTS_DIR/${CONFIG}_seed${SEED}-env=${ENV_NAME}.txt
    cp ./tmp/DDQN_${SEED}.txt $RESULT_FILE
done