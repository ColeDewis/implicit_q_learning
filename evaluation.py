from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        i = 0
        while not done:
            print(observation)
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)

            if i > 500:
                done = True

            i += 1

        print(info)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
