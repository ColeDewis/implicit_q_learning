import os
import random
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from env_client import RemoteRLBenchEnv
import wrappers
from dataset_utils import D4RLDataset, RLBenchDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
from DDQN_learner import DDQNLearner
import jax

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 2, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("max_approx_method", "CEM", "Method to use for approximating max.")
# flags.DEFINE_float("tau", 0.8, "Controls how fast target networks are changed.")
flags.DEFINE_string("learner", "IQL", "Learning algorithm to use ('IQL' or 'DDQN').")
flags.DEFINE_integer("port", 5000, "port for communicating with rlbench")
config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_list(
    "overrides", None, "List of hyperparameter overrides in the format key=value."
)

def normalize(dataset):

    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str, seed: int, is_d4rl: bool) -> Tuple[gym.Env, D4RLDataset | RLBenchDataset]:
    # NOTE I think this has to be before making env
    np.random.seed(seed)
    random.seed(seed)

    if is_d4rl:
        env = gym.make(env_name, seed=seed)
        dataset = D4RLDataset(env)
    else:
        dataset = RLBenchDataset('microwave_data.h5')
        env = RemoteRLBenchEnv(FLAGS.port)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)

    # NOTE: antmaze doesn't seed properly otherwise, this seems to resolve it
    # see: https://github.com/Farama-Foundation/D4RL/issues/202
    # here we directly pull out the AntMazeEnv to call its seed method.
    if "antmaze" in FLAGS.env_name:
        # TODO test on cc.
        tempenv = env
        while hasattr(tempenv, "env"):
            tempenv = tempenv.env
        tempenv._wrapped_env.seed(seed)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)


    if "antmaze" in FLAGS.env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif (
        "halfcheetah" in FLAGS.env_name
        or "walker2d" in FLAGS.env_name
        or "hopper" in FLAGS.env_name
    ):
        normalize(dataset)

    return env, dataset


def apply_overrides(config, overrides):
    """Apply a list of key=value overrides to the config."""
    if overrides:
        for override in overrides:
            key, value = override.split("=")
            # Convert value to the appropriate type
            if value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            elif value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            config[key] = value


def main(_):
    print(f"JAX default backend: {jax.default_backend()}")
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, "tb", str(FLAGS.seed)), write_to_disk=True
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    if ("antmaze" in FLAGS.env_name
        or "halfcheetah" in FLAGS.env_name
        or "walker2d" in FLAGS.env_name
        or "hopper" in FLAGS.env_name
    ):
        is_d4rl = True
    else:
        is_d4rl = False


    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, is_d4rl)

    kwargs = dict(FLAGS.config)
    apply_overrides(kwargs, FLAGS.overrides)
    save_file_name = f"{FLAGS.learner}_{FLAGS.seed}.txt"
    if FLAGS.learner == "DDQN":
        # save_file_name = f"{FLAGS.learner}_{FLAGS.max_approx_method}_{FLAGS.seed}.txt"
        agent = DDQNLearner(
            FLAGS.seed,
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            max_steps=FLAGS.max_steps,
            **kwargs,
        )
    elif FLAGS.learner == "IQL":
        # save_file_name = f"{FLAGS.learner}_{FLAGS.seed}.txt"
        agent = Learner(
            FLAGS.seed,
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            max_steps=FLAGS.max_steps,
            **kwargs,
        )
    else:
        assert(f"Learner {FLAGS.learner} is not implemented")

    eval_returns = []
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        batch = dataset.sample(FLAGS.batch_size)

        # update target
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f"training/{k}", v, i)
                else:
                    summary_writer.add_histogram(f"training/{k}", v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats["return"]))
            np.savetxt(
                os.path.join(FLAGS.save_dir, save_file_name),
                eval_returns,
                fmt=["%d", "%.1f"],
            )


if __name__ == "__main__":
    app.run(main)
