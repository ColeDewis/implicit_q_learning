from random import randint, uniform
from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params

# fmt: off

from max_approx.CrossEntropy import CEM


def get_max_actions_values(critic: Model, states: Tuple, num_actions: int):
    cem = CEM(critic, d=num_actions, maxits=2, N=10, Ne=5)

    best_actions = [cem.eval(state) for state in states]
    q_values = [critic(state, best_action) for state, best_action in zip(states, best_actions)]

    return jnp.array(best_actions), jnp.array(q_values)


    #                          best actions                                                                           Value of best actions
    # return jnp.array([jnp.array([uniform(*[0, 5])]*8) for i in range(len(states))]).reshape(-1, 8), jnp.array([uniform(*[0, 5]) for i in range(len(states))])

def update_v(critic: Model, value: Model, batch: Batch, max_action_values) -> Tuple[Model, InfoDict]:
    # actions = batch.actions

    _, q = get_max_actions_values(critic, batch.observations, batch.actions.shape[1])

    
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
             discount: float) -> Tuple[Model, InfoDict]:
    
    max_actions, _ = get_max_actions_values(critic, batch.observations, batch.actions.shape[1])

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
