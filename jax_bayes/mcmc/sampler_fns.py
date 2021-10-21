"""Sampler functions for use with jax-bayes."""

import math

import jax
import jax.numpy as jnp
from jax.experimental.optimizers import make_schedule

from .sampler import sampler
from .utils import init_distributions

def match_dims(src, target, start_dim=1):
    """
    returns an array with the data from 'src' and same number of dims
    as 'target' by padding with empty dimensions starting at 'start_dim'
    """
    new_dims = tuple(range(start_dim, len(target.shape)))
    return jnp.expand_dims(src, new_dims)

@sampler
def langevin_fns(base_key, num_samples=10, step_size=1e-3, init_stddev=0.1,
                 init_dist='normal'):
    """Constructs sampler functions for the Unadjusted Langevin Algorithm.
    See e.g. https://arxiv.org/pdf/1605.01559.pdf.

    Args:
        base_key: jax.random.PRNGKey
        num_samples: number of samples to initialize; either > 0 or -1.
            If num_samples == -1, assumes that the initial samples are
            already constructed.
        step_size: float or callable w/ signature step_size(i)
        init_stdev: nonnegative float standard deviation for initialization
            of initial samples (ignored if num_samples == -1)
        init_dist: str in ['normal', 'centered_uniform'] to sample perturbations
            for the initial distribution

    Returns:
        sampler function tuple (init, propose, update, get_params)
    """
    step_size = make_schedule(step_size)

    if isinstance(init_dist, str):
        init_dist = init_distributions[init_dist]

    def init(x0, key):
        init_key, next_key = jax.random.split(key)
        if num_samples == -1:
            return x0, next_key
        x = init_dist(init_key, (num_samples,) + x0.shape)
        return x0 + init_stddev * x, next_key

    def propose(i, g, x, key):
        key, next_key = jax.random.split(key)
        Z = jax.random.normal(key, x.shape)
        return x + step_size(i) * g + math.sqrt(2 * step_size(i)) * Z, next_key

    def update(i, logp_x, logp_xprop, g, x, gprop, xprop, key):
        key, next_key = jax.random.split(key)
        return xprop, next_key

    def get_params(x):
        return x

    return init, propose, update, get_params, base_key

@sampler
def mala_fns(base_key, num_samples=10, step_size=1e-3,
             init_stddev=0.1, init_dist='normal'):
    """Constructs sampler functions for the Metropolis Adjusted Langevin Algorithm.
    See e.g. http://probability.ca/jeff/ftpdir/lang.pdf

    Args:
        base_key: jax.random.PRNGKey
        num_samples: number of samples to initialize; either > 0 or -1.
            If num_samples == -1, assumes that the initial samples are
            already constructed.
        step_size: float or callable w/ signature step_size(i)
        init_stdev: nonnegative float standard deviation for initialization
            of initial samples (ignored if num_samples == -1)
        init_dist: str in ['normal', 'centered_uniform'] to sample perturbations
            for the initial distribution

    Returns:
        sampler function tuple (init, propose, update, get_params)
    """
    step_size = make_schedule(step_size)

    if isinstance(init_dist, str):
        init_dist = init_distributions[init_dist]

    def log_proposal(i, y, x, dx):
        #computes log q(y|x)
        return - 0.5 * jnp.sum(jnp.square((y - x - step_size(i) * dx)) \
                / math.sqrt(2 *step_size(i)))
    log_proposal = jax.vmap(log_proposal, in_axes=(None, 0, 0, 0))

    def init(x0, key):
        init_key, next_key = jax.random.split(key)
        if num_samples == -1:
            return x0, next_key
        x = init_dist(init_key, (num_samples,) + x0.shape)
        return x0 + init_stddev * x, next_key

    def propose(i, g, x, key):
        key, next_key = jax.random.split(key)
        Z = jax.random.normal(key, x.shape)
        return x + step_size(i) * g + math.sqrt(2 * step_size(i)) * Z, next_key

    def update(i, logp_x, logp_xprop, g, x, gprop, xprop, key):
        key, next_key = jax.random.split(key)
        U = jax.random.uniform(key, (x.shape[0],))

        log_alpha = (logp_xprop + log_proposal(i, x, xprop, gprop)
                     - logp_x - log_proposal(i, xprop, x, g))

        accept_idxs = jnp.log(U) < log_alpha
        accept_idxs = match_dims(accept_idxs, x)
        mask = accept_idxs.astype(jnp.float32)

        xnext = x * (1.0 - mask) + xprop * mask
        return xnext, next_key

    def get_params(x):
        return x

    return init, propose, update, get_params, base_key

@sampler
def rk_langevin_fns(base_key, num_samples=10, step_size=1e-3,
                    init_stddev=0.1, init_dist='normal'):
    """Constructs sampler functions for a Stochastic Runge Kutta integrator of the
    continuous-time Langevin dynamics.

    See e.g. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE).

    One step of the integration is computed as 0.5*(K1 + K2) where K1, K2
    are the two 'knots' of the integrator. K1 is computed in propose(...) and
    K2 is computed in update(...) since we need to re-compute gradients.

    Args:
        base_key: jax.random.PRNGKey
        num_samples: number of samples to initialize; either > 0 or -1.
            If num_samples == -1, assumes that the initial samples are
            already constructed.
        step_size: float or callable w/ signature step_size(i)
        init_stdev: nonnegative float standard deviation for initialization
            of initial samples (ignored if num_samples == -1)
        init_dist: str in ['normal', 'centered_uniform'] to sample perturbations
            for the initial distribution

    Returns:
        sampler function tuple (init, propose, update, get_params)
    """
    step_size = make_schedule(step_size)

    if isinstance(init_dist, str):
        init_dist = init_distributions[init_dist]


    def init(x0, key):
        init_key, next_key = jax.random.split(key)
        if num_samples == -1:
            return x0, next_key
        x = init_dist(init_key, (num_samples,) + x0.shape)
        return x0 + init_stddev * x, next_key

    def propose(i, g, x, key):
        h = step_size(i)
        root_h = math.sqrt(h)

        w_key, s_key, next_key = jax.random.split(key, 3)
        W = jax.random.normal(w_key, x.shape) * root_h
        S = jax.random.bernoulli(s_key, 0.5, (x.shape[0],)) * 2 - 1
        S = match_dims(S, x)

        K1 = x + h * g + (W - root_h * S) * math.sqrt(2)

        return K1, key

    def update(i, logp_x, logp_xprop, g, x, gprop, xprop, key):
        h = step_size(i)
        root_h = math.sqrt(h)

        w_key, s_key, next_key = jax.random.split(key, 3)
        W = jax.random.normal(w_key, x.shape) * root_h
        S = jax.random.bernoulli(s_key, 0.5, (x.shape[0],)) * 2 - 1
        S = match_dims(S, x)

        K2 = h * gprop + (W + root_h * S) * math.sqrt(2)
        return  0.5 * x + 0.5 * (xprop + K2), next_key

    def get_params(x):
        return x

    return init, propose, update, get_params, base_key

@sampler
def hmc_fns(base_key, num_samples=10, step_size=1e-3,
            init_stddev=0.1, init_dist='normal'):
    """Constructs sampler functions for the Hamiltonia Monte Carlo algorithm.
    See e.g. http://probability.ca/jeff/ftpdir/lang.pdf

    Args:
        base_key: jax.random.PRNGKey
        num_samples: number of samples to initialize; either > 0 or -1.
            If num_samples == -1, assumes that the initial samples are
            already constructed.
        step_size: float or callable w/ signature step_size(i)
        init_stdev: nonnegative float standard deviation for initialization
            of initial samples (ignored if num_samples == -1)
        init_dist: str in ['normal', 'centered_uniform'] to sample perturbations
            for the initial distribution

    Returns:
        sampler function tuple (init, propose, update, get_params)
    """
    step_size = make_schedule(step_size)

    if isinstance(init_dist, str):
        init_dist = init_distributions[init_dist]

    @jax.vmap
    def dot_product(x, y):
        return jnp.sum(x * y)

    def init(x0, key):
        init_key, R_key, next_key = jax.random.split(key, 3)
        if num_samples == -1:
            R = jax.random.normal(R_key, x0.shape)
            return (x0, R), next_key
        x = init_dist(init_key, (num_samples,) + x0.shape)
        R = jax.random.normal(R_key, (num_samples,) + x0.shape)
        return (x0 + init_stddev * x, R), next_key

    def propose(i, g, x, key):
        """
        iterate this several times for multistep leapfrog integrator. Note
        the final update for rprop is in the update
        """
        next_key = key
        x, r = x
        rprop = r + 0.5 * step_size(i) * g
        xprop = x + step_size(i) * rprop
        return (xprop, rprop), next_key

    def update(i, logp_x, logp_xprop, g, x, gprop, xprop, key):
        u_key, r_key, next_key = jax.random.split(key, 3)

        x, r = x
        xprop, rprop = xprop
        #this is the last step of the leapfrog integrator
        rprop = rprop + 0.5 * step_size(i) * gprop

        U = jax.random.uniform(key, (x.shape[0],))

        log_alpha = (logp_xprop + dot_product(rprop, rprop)
                     - logp_x - dot_product(r, r))

        accept_idxs = jnp.log(U) < log_alpha

        accept_idxs = match_dims(accept_idxs, x)
        mask = accept_idxs.astype(jnp.float32)

        xnext = x * (1.0 - mask) + xprop * mask

        #this is for the first step of the leapfrog integrator
        rnext = jax.random.normal(r_key, x.shape)
        return (xnext, rnext), next_key

    def get_params(x):
        return x[0]

    return init, propose, update, get_params, base_key


@sampler
def rms_langevin_fns(base_key, num_samples=10, step_size=1e-3, beta=0.9,
                     eps=1e-9, init_stddev=0.1, init_dist='normal'):
    """Constructs sampler functions for the RMS-preconditioned Langevin algorithm.
    See e.g. https://arxiv.org/pdf/1512.07666.pdf

    Args:
        base_key: jax.random.PRNGKey
        num_samples: number of samples to initialize; either > 0 or -1.
            If num_samples == -1, assumes that the initial samples are
            already constructed.
        step_size: float or callable w/ signature step_size(i)
        init_stdev: nonnegative float standard deviation for initialization
            of initial samples (ignored if num_samples == -1)
        init_dist: str in ['normal', 'centered_uniform'] to sample perturbations
            for the initial distribution

    Returns:
        sampler function tuple (init, propose, update, get_params)
    """
    step_size = make_schedule(step_size)

    if isinstance(init_dist, str):
        init_dist = init_distributions[init_dist]

    def init(x0, key):
        init_key, next_key = jax.random.split(key)
        if num_samples == -1:
            r = jnp.zeros_like(x0)
            return (x0, r), next_key
        x = init_dist(init_key, (num_samples,) + x0.shape)
        r = jnp.zeros_like(x)
        return (x0 + init_stddev * x, r), next_key

    def propose(i, g, x, key):
        key, next_key = jax.random.split(key)
        x, r = x
        Z = jax.random.normal(key, x.shape)

        r = beta * r + (1. - beta) * jnp.square(g)

        ss = step_size(i) / (jnp.sqrt(r) + eps)
        xprop = x + ss * g + jnp.sqrt(2 * ss) * Z

        return (xprop, r), next_key

    def update(i, logp_x, logp_xprop, g, x, gprop, xprop, key):
        key, next_key = jax.random.split(key)
        return xprop, next_key

    def get_params(x):
        x, _ = x
        return x

    return init, propose, update, get_params, base_key


@sampler
def rwmh_fns(base_key, num_samples=10, step_size=1e-3,
             init_stddev=0.1, init_dist='normal'):
    """Constructs sampler functions for Random Walk Metropolis Hastings.
    See e.g. https://arxiv.org/pdf/1504.01896.pdf

    Args:
        base_key: jax.random.PRNGKey
        num_samples: number of samples to initialize; either > 0 or -1.
            If num_samples == -1, assumes that the initial samples are
            already constructed.
        step_size: float or callable w/ signature step_size(i)
        init_stdev: nonnegative float standard deviation for initialization
            of initial samples (ignored if num_samples == -1)
        init_dist: str in ['normal', 'centered_uniform'] to sample perturbations
            for the initial distribution

    Returns:
        sampler function tuple (init, propose, update, get_params)
    """
    step_size = make_schedule(step_size)

    if isinstance(init_dist, str):
        init_dist = init_distributions[init_dist]

    def init(x0, key):
        init_key, next_key = jax.random.split(key)
        if num_samples == -1:
            return x0, next_key
        x = init_dist(init_key, (num_samples,) + x0.shape)
        return x0 + init_stddev * x, next_key

    def propose(i, g, x, key):
        key, next_key = jax.random.split(key)
        Z = jax.random.normal(key, x.shape)
        return x + step_size(i) * Z, next_key

    def update(i, logp_x, logp_xprop, g, x, gprop, xprop, key):
        key, next_key = jax.random.split(key)
        U = jax.random.uniform(key, (x.shape[0],))

        log_alpha = logp_xprop - logp_x

        accept_idxs = jnp.log(U) < log_alpha
        accept_idxs = match_dims(accept_idxs, x)
        mask = accept_idxs.astype(jnp.float32)

        xnext = x * (1.0 - mask) + xprop * mask
        return xnext, next_key

    def get_params(x):
        return x

    return init, propose, update, get_params, base_key
