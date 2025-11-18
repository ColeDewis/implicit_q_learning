from random import randint, uniform
from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params

# fmt: off

from max_approx.CrossEntropy import CEM


def update_v(critic: Model, value: Model, batch: Batch, max_action_values) -> Tuple[Model, InfoDict]:    
    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        # this may be invalid, might need to go and get the prob of each s,a pair
        value_loss = (0.5 * (v - max_action_values)**2).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info

def update_q(critic: Model, target_critic:Model, batch: Batch, discount: float, max_actions) -> Tuple[Model, InfoDict]:
    next_actions = max_actions
    next_action_values = target_critic(batch.next_observations, next_actions)

    target_q = batch.rewards + discount * batch.masks * next_action_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:

        q_val = critic.apply({'params': critic_params}, batch.observations, batch.actions)

        critic_loss = ((q_val - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_val': q_val.mean(),
        }
    
    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
