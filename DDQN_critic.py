from typing import Tuple

import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params

from random import randint


def get_max_actions(critic: Model, states: Tuple, action_range: Tuple):
    return [randint(*action_range) for i in range(len(states))]

def update_v(critic: Model, value: Model, batch: Batch,
             action_space: jnp.array) -> Tuple[Model, InfoDict]:
    # actions = batch.actions
    all_qs = []
    for act in action_space:
        all_qs.append(critic(batch.observations, [act] * len(batch.observations)))
    
    q_means = jnp.mean(all_qs, axis=0)

    
    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        assert(q_means.shape == v.shape, f"Jasper fucked up the q_mean stuff\nq_mean shape: {q_means.shape}\nv shape: {v.shape}")
        # this may be invalid, might need to go and get the prob of each s,a pair
        value_loss = value_loss = 0.5 * (v - q_means)**2
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
    
    next_actions = get_max_actions(critic, batch.observations, [0, 1])

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
