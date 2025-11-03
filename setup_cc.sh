#!/bin/bash
START_DIR=$(pwd)

mkdir ~/.mujoco
cd ~/.mujoco && wget https://www.roboti.us/download/mjpro150_linux.zip && wget https://www.roboti.us/file/mjkey.txt && unzip mjpro150_linux.zip 

# referenced from https://github.com/andnp/rl-control-template/blob/main/scripts/setup_cc.sh
echo "scheduling a job to install project dependencies"

# seems to need a lot of mem to install the requirements.
cd $START_DIR
sbatch --ntasks=1 --mem-per-cpu="16G" --export=path="$(pwd)" venv_setup.sh