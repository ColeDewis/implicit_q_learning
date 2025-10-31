#!/bin/bash
# NOTE this is untested and referenced from https://github.com/andnp/rl-control-template/

#SBATCH --time=00:55:00

module load python/3.10
module load mujoco/3.1.6
module load cuda
export MUJOCO_PATH=~/.mujoco/mjpro150
export MUJOCO_PLUGIN_PATH=~/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin/:$LD_LIBRARY_PATH

cp $path/requirements_cc.txt $SLURM_TMPDIR/
cd $SLURM_TMPDIR
python -m venv .venv
source .venv/bin/activate
pip install "cython<3"
pip install -r requirements_cc.txt

tar -cavf venv310.tar .venv
cp venv310.tar $path/

pip freeze