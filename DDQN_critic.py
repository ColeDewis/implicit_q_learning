from random import randint, uniform
from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params

# fmt: off

def get_max_actions_values(critic: Model, states: Tuple, action_dim: int, action_range: Tuple):
    action_dim = int(action_dim)
    #                          best actions                                                                           Value of best actions
    return jnp.array([jnp.array([uniform(*action_range)]*action_dim) for i in range(len(states))]).reshape(-1, action_dim), jnp.array([uniform(*action_range) for i in range(len(states))])

def update_v(critic: Model, value: Model, batch: Batch, max_action_values) -> Tuple[Model, InfoDict]:
    # actions = batch.actions
    # _, q = get_max_actions_values(critic, batch.observations, action_dim, [0, 5])
    q = max_action_values
    
    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        # this may be invalid, might need to go and get the prob of each s,a pair
        value_loss = (0.5 * (v - q)**2).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info
    ###############################################

    # q1, q2 = critic(batch.observations, actions)
    # q = jnp.minimum(q1, q2)

    # def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
    #     v = value.apply({'params': value_params}, batch.observations)
    #     value_loss = loss(q - v, expectile).mean()
    #     return value_loss, {
    #         'value_loss': value_loss,
    #         'v': v.mean(),
    #     }

    # new_value, info = value.apply_gradient(value_loss_fn)

    # return new_value, info

def update_q(critic: Model, target_critic:Model, target_value: Model, batch: Batch,
             discount: float, max_actions) -> Tuple[Model, InfoDict]:

    # next_actions, _ = get_max_actions_values(critic, batch.observations, action_dim, [0, 5])

    next_actions = max_actions
    next_action_values = target_critic(batch.observations, next_actions)

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
    ##############################################################
    # next_v = target_value(batch.next_observations)

    # target_q = batch.rewards + discount * batch.masks * next_v

    # def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
    #     q1, q2 = critic.apply({'params': critic_params}, batch.observations,
    #                           batch.actions)
    #     critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
    #     return critic_loss, {
    #         'critic_loss': critic_loss,
    #         'q1': q1.mean(),
    #         'q2': q2.mean()
    #     }

    # new_critic, info = critic.apply_gradient(critic_loss_fn)

    # return new_critic, info
