from typing import Tuple
import jax
import jax.numpy as jnp
from common import Model

class ActionGradientAscent:
        def __init__(self, critic: Model, d: int, maxits: int = 20, action_range: Tuple[float, float] = (-1.0, 1.0), step_size: float = 0.1):
            self.critic = critic
            self.d = d
            self.maxits = maxits # number of GA steps
            self.lb, self.ub = action_range # action bounds (ENVIRONMENT SPECIFIC!!)
            self.step_size = step_size

        def q(self, actions: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
            ''' Get a single q value for a state and action. Only care about a single state and action, as eval gets one state only since action gradient ascent is per state '''
            q = self.critic.apply({'params': self.critic.params}, states[None, ...], actions[None, ...]) # must batch 
            return jnp.squeeze(q)
        

        def ascent(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            ''' Perform gradient ascent on Q func wrt a '''


            a0 = jnp.full((self.d,), 0.5 * (self.lb + self.ub)) # initial action (midpoint of bounds)

            # inner func because no python loops allowed for jax jit i guess
            def inner_func(i, a):
                grad_a = jax.grad(self.q)(a, state) # get gradient of q func wrt a, (jax moves gradient function for q outside loop automatically, and this just plugs in each new a?)
                a = a + self.step_size * grad_a # gradient ascent step
                return jnp.clip(a, self.lb, self.ub) # keep action in bounds
            
            best_action = jax.lax.fori_loop(0, self.maxits, inner_func, a0)
            best_q = self.q(best_action, state)
            return best_action, best_q
        
        def eval(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
                return self.ascent(state)






