import time
from tqdm import tqdm

import jax
from jax import grad, value_and_grad


centered_uniform = \
    lambda *args, **kwargs: jax.random.uniform(*args, **kwargs) - 0.5
init_distributions = dict(normal=jax.random.normal,
                          uniform=centered_uniform)

#redefine elementwise_grad operation
elementwise_grad = lambda f: jax.vmap(grad(f))
elementwise_value_and_grad = lambda f: jax.vmap(value_and_grad(f))

def blackbox_mcmc(
    logprob,
    x0,
    sampler_fn,
    num_iters=1000,
    proposal_iters=1,
    seed=None,
    recompute_grad=False,
    use_jit=True,
    **sampler_args
):
    """ A single-function black-box sampler abstracting the various pieces
    of the functional sampler methodologies.

    Args:
        logprob: a callable logbrob(x) that returns the unnormalized
            log probability of x.
        x0: array of initial sample(s)
        sampler_fn: a sampler_fn using the @sampler decorator (se sampler.py)
        num_iters: number of iterations
        proposal_iters: number of times to compute proposal w/ new gradients
        seed: seed for the keys
        recompute_grad: boolean for whether to recompute the gradients (use if
            proposal_iters > 0)
        use_jit: boolean for jitting the update step (for debugging).

    Returns:
        approximate samples according to logprob using the sampler_fn.
    """

    g = elementwise_value_and_grad(logprob)

    seed = int(time.time() * 1000) if seed is None else seed
    init_key = jax.random.PRNGKey(seed)

    sampler = sampler_fn(init_key, **sampler_args)
    sampler_state, sampler_keys = sampler.init(x0)

    def _step(i, state, keys):
        x = sampler.get_params(state)
        fx, dx = g(x)

        prop_state, keys = sampler.propose(i, dx, state, keys)
        x_prop = sampler.get_params(prop_state)
        if recompute_grad:
            fx_prop, dx_prop = g(x_prop)
            for _ in range(max(proposal_iters-1, 0)):
                prop_state, keys = sampler.propose(i, dx_prop, prop_state, keys)
                x_prop = sampler.get_params(prop_state)
                fx_prop, dx_prop = g(x_prop)
            if proposal_iters > 1:
                prop_state, keys = sampler.propose(i, dx_prop, prop_state, keys, is_final=True)
                x_prop = sampler.get_params(prop_state)
                fx_prop, dx_prop = g(x_prop)
        else:
            fx_prop, dx_prop = fx, dx

        accept_idxs, keys = sampler.accept(
            i, fx, fx_prop, dx, state, dx_prop, prop_state, keys
        )

        state, keys = sampler.update(
            i, accept_idxs, dx, state, dx, prop_state, keys
        )
        return state, keys

    if use_jit:
        _step = jax.jit(_step)

    for i in tqdm(range(num_iters)):
        # if callback: callback(x, i, dx)
        sampler_state, sampler_keys = _step(i, sampler_state, sampler_keys)

    return sampler.get_params(sampler_state)
