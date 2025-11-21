# Offline Reinforcement Learning with Implicit Q-Learning

This repository contains the official implementation of [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) by [Ilya Kostrikov](https://kostrikov.xyz), [Ashvin Nair](https://ashvin.me/), and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).

If you use this code for your research, please consider citing the paper:
```
@article{kostrikov2021iql,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

For a PyTorch reimplementation see https://github.com/rail-berkeley/rlkit/tree/master/examples/iql

## How to run the code

**This was tested using python 3.10**

Go to:
https://www.roboti.us/download.html
https://www.roboti.us/license.html

and download mjpro150, put in .mujoco folder. Put the license there also. 

Install depenedencies: `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3` and `sudo apt-get install patchelf`. 

Then run `pip install "cython<3"`

### Install dependencies

```bash
pip install -r requirements_cuda12.txt

```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Run training

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000
```

Kitchen and Adroit
```bash
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py
```

Finetuning on AntMaze tasks
```bash
python train_finetune.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_finetune_config.py --eval_episodes=100 --eval_interval=100000 --replay_buffer_size 2000000
```

## Compute Canada Setup

**Below is tested on fir, nibi, narval**

Use the `requirements_cc.txt` file. 

To set up dependencies, clone the repo, then inside the folder:
`chmod +x setup_cc.sh`
`chmod +x venv_setup.sh`
`./setup_cc.sh`

Then, you need to create the virtual env. On nibi and fir, I used an interactive job:

`salloc --time=0:30:0 --mem-per-cpu=16G --ntasks=1`

to try and make things faster. On other clusters (e.g. narval) I believe the compute nodes don't have internet access; on there I just ran it on the login node.

Then run `./venv_setup.sh` to create the `venv310.tar` file.

Once you have `venv310.tar` in your folder, try the test job. **Note that the GPU you need to specify may vary between clusters**. See https://docs.alliancecan.ca/wiki/Multi-Instance_GPU#Available_configurations

Nibi and fir: `#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1`
Narval: `#SBATCH --gpus-per-node=a100_3g.20gb:1`

The test job does require the dataset to already be on the compute canada server. You can download in advance from here: https://huggingface.co/datasets/imone/D4RL/tree/main, since I think it's better to get the dataset in advance rather than having your job download it. You need `Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5`. Put it in a folder `~/.d4rl/datasets/` by copying it to the cluster with `scp`.

Run the test job with `sbatch --export=path="$(pwd)" job_scripts/test_job.sh`. 


### Notes about this setup
I tried having the virtual env be created in a non-interactive job, but it seemed to have issues with using the venv after.




## Misc
The implementation is based on [JAXRL](https://github.com/ikostrikov/jaxrl).
