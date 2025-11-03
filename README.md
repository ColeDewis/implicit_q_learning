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

Use the `requirements_cc.txt` file. 

To set up dependencies, clone the repo, then inside the folder:
`chmod +x setup_cc.sh`
`./setup_cc.sh`

This will schedule a job to create a virtual environment tarfile. Once you have `venv310.tar` in your folder, try the test job.

Run the test job with `sbatch --export=path="$(pwd)" test_job.sh`

I'm not sure if this process works on other clusters since one of the requirements requires a git clone, and supposedly the compute nodes don't have internet access for narval and others.  

The test job does require the dataset to already be on the compute canada server. You can download in advance from here: https://huggingface.co/datasets/imone/D4RL/blob/main, since I think it's better to get the dataset in advance rather than having your job download it.

**You can try the setup_cc.sh and venv_setup.sh scripts but they are untested so far**, as I followed those steps in an interactive job.

The `test_job.sh` script should work however.

## Misc
The implementation is based on [JAXRL](https://github.com/ikostrikov/jaxrl).
