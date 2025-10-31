#!/bin/bash
mkdir ~/.mujoco
cd ~/.mujoco && wget https://mujoco.org/download/mjpro150_linux.zip && wget https://mujoco.org/download/mjkey.txt && unzip mjpro150_linux.zip 

# referenced from https://github.com/andnp/rl-control-template/blob/main/scripts/setup_cc.sh
echo "scheduling a job to install project dependencies"

# seems to need a lot of mem to install the requirements.
sbatch --ntasks=1 --mem-per-cpu="16G" --export=path="$(pwd)" scripts/venv_setup.sh