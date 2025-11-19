"""Implementations of algorithms for continuous control."""

import time
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from common import Batch, InfoDict, Model, PRNGKey
from DDQN_actor import update as awr_update_actor
from DDQN_critic import update_q, update_v

from functools import partial

from max_approx.CrossEntropy import CEM 

# fmt: off

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)

# TODO: Also need to try CEM, gradient ascent. Also, should try sampling action rather than
# mode possibly.
def get_max_actions_values_AC(actor: Model, critic: Model, states: jnp.ndarray, temperature: float):
    mode_actions = policy.sample_actions_deterministic(actor.apply_fn,
                                             actor.params, states,
                                             temperature)
    mode_action_values = critic.apply({'params': critic.params}, states, mode_actions)

    return mode_actions, mode_action_values

def get_max_actions_values_CEM(critic: Model, states: Tuple, num_actions: int, CEM_rng: PRNGKey):
    cem = CEM(critic, d=num_actions, maxits=2, N=10, Ne=5, rand_key=CEM_rng)

    best_actions = jnp.array([cem.eval(state) for state in states])
    q_values = critic(states, best_actions)

    return best_actions, q_values


# @jax.jit
@partial(jax.jit, static_argnames=['max_approx_method'])
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    temperature: float, max_approx_method:str
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)

    # TODO: not sure whether to use critic or target critic here to be "more correct"
    if max_approx_method == "CEM":
        # TODO: not sure if this messes with other rand seeds later, or if that even matters
        CEM_rng, rng = jax.random.split(rng)
        max_actions, max_action_values = get_max_actions_values_CEM(critic, batch.next_observations, batch.actions.shape[1], CEM_rng)
    elif max_approx_method == "AC":
        max_actions, max_action_values = get_max_actions_values_AC(actor, critic, batch.next_observations, temperature)


    new_value, value_info = update_v(target_critic, value, batch, max_action_values)

    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, batch, temperature)
    
    # TODO: I run this again since the actor has been updated. I'm not sure if we should keep this order
    # of updates or not though.
    # max_actions, _ = get_max_actions_values_actor(actor, critic, batch.next_observations, temperature)

    new_critic, critic_info = update_q(critic, target_critic, batch, discount, max_actions)
    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


class DDQNLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine", 
                 max_approx_method: str = "CEM"):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.discount = discount
        self.tau = tau
        self.temperature = temperature

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = value_net.Critic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = value_net.ValueCritic(hidden_dims)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng
        self.action_space = actions
        self.action_dim = action_dim
        self.max_approx_method = max_approx_method


    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch, self.discount, self.tau, self.temperature, self.max_approx_method)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info
