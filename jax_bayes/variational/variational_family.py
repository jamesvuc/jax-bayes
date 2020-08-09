from collections import namedtuple
import functools

import jax
from jax.util import partial, safe_zip, safe_map, unzip2
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node

from ..mcmc.sampler import SamplerKeys, SamplerState

map = safe_map
zip = safe_zip

VariationalParams = namedtuple(
    "VariationalParams",
    ["packed_state", "tree_def", "subtree_defs"]
)
register_pytree_node(
    VariationalParams,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs: VariationalParams(xs[0], data[0], data[1])
)

class VariationalFamily:
    init = None
    sample = None
    evaluate = None
    get_samples = None
    get_params = None
    get_params = None
    next_key = None
    entropy = None

def variational_family(var_maker):
    """Decorator to make an optimizer defined for arrays generalize to containers.

    With this decorator, you can write variational_family functions that
    each operate only on single arrays, and convert them to corresponding
    functions that operate on pytrees of parameters. See the optimizers defined in
    optimizers.py for examples.

    Note: The variational families produced by this function are limited to modelling
    (at most) block-diagonal dependence with one block per leaf node in the pytree.
    This is used when we sum the log-probabilities (e.g. in tree_evaluate).

    Args:
        var_maker: a function that returns an ``(init, sample, evaluate, get_samples,
        next_key, entropy, init_key)`` tuple of functions that might only work
        with ndarrays.

    Returns:
        A ``VariationalFamily object`` that collects tree-ified versions of the
        above functions that work on arbitrary pytrees.

        The VariationalParams pytree type used by the returned functions is isomorphic
        to ``ParameterPytree (OptStatePytree ndarray)``, but may store the state
        instead as e.g. a partially-flattened data structure for performance.
    """

    @functools.wraps(var_maker)
    def tree_var_maker(*args, **kwargs):
        init, sample, evaluate, get_samples, next_key, entropy, init_key = \
            var_maker(*args, **kwargs)

        @functools.wraps(init)
        def tree_init(x0_tree):
            x0_flat, tree = tree_flatten(x0_tree)
            initial_keys = jax.random.split(init_key, len(x0_flat))
            initial_params, initial_keys = unzip2(init(x0, k) for x0, k in \
                                                  zip(x0_flat, initial_keys))
            params_flat, subtrees = unzip2(map(tree_flatten, initial_params))
            return VariationalParams(params_flat, tree, subtrees), SamplerKeys(initial_keys)

        @functools.wraps(sample)
        def tree_sample(i, num_samples, tree_keys, var_params):
            params_flat, tree, subtrees = var_params
            params = map(tree_unflatten, subtrees, params_flat)
            keys, keys_meta = tree_flatten(tree_keys)
            samples, new_keys = unzip2(map(partial(sample, i, num_samples), keys, params))
            samples_flat, subtrees2 = unzip2(map(tree_flatten, samples))
            return SamplerState(samples_flat, tree, subtrees2), SamplerKeys(new_keys)

        @functools.wraps(evaluate)
        def tree_evaluate(inputs, var_params):
            """ this assumes each factor is independent (i.e. block-diagonal) """
            params_flat, tree, subtrees = var_params
            params = map(tree_unflatten, subtrees, params_flat)

            #inputs is a also a pytree with the same structure as var_params
            inputs_flat, tree2, subtrees2 = inputs
            inputs = map(tree_unflatten, subtrees2, inputs_flat)

            if tree2 != tree:
                msg = ("evaluate update function was passed a inputs tree that did "
                       "not match the parameter tree structure with which it was "
                       "initialized: parameter tree {} and inputs tree {}.")
                raise TypeError(msg.format(tree, tree2))

            logprob = sum(evaluate(x, p) for x, p in zip(inputs, params))
            return logprob

        @functools.wraps(get_samples)
        def tree_get_samples(var_state):
            states_flat, tree, subtrees = var_state
            states = map(tree_unflatten, subtrees, states_flat)
            samples = map(get_samples, states)
            return tree_unflatten(tree, samples)

        tree_get_params = tree_get_samples

        @functools.wraps(next_key)
        def tree_next_key(tree_keys):
            keys, keys_meta = tree_flatten(tree_keys)
            new_keys = [next_key(key) for key in keys]
            return SamplerKeys(new_keys)

        @functools.wraps(entropy)
        def tree_entropy(var_params):
            params_flat, tree, subtrees = var_params
            params = map(tree_unflatten, subtrees, params_flat)

            ent = sum(entropy(p) for p in params)
            return ent

        var_family = VariationalFamily()
        var_family.init = tree_init
        var_family.sample = tree_sample
        var_family.evaluate = tree_evaluate
        var_family.get_samples = tree_get_samples
        var_family.get_params = tree_get_params
        var_family.next_key = tree_next_key
        var_family.entropy = tree_entropy

        return var_family

    return tree_var_maker
