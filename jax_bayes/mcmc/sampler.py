from collections import namedtuple
import functools

import jax
import jax.numpy as jnp
from jax._src.util import partial, safe_zip, safe_map, unzip2
from jax._src.tree_util import (
    tree_flatten,
    tree_unflatten,
    register_pytree_node,
)

from jax.tree_util import tree_map, tree_leaves


map = safe_map
zip = safe_zip

def sum_tree_leaves(tree):
    return sum(leaf for leaf in tree_leaves(tree))


# from jax.experimental.optimizers.py:
# The implementation here basically works by flattening pytrees. There are two
# levels of pytrees to think about: the pytree of params, which we can think of
# as defining an "outer pytree", and a pytree produced by applying init_fun to
# each leaf of the params pytree, which we can think of as the "inner pytrees".
# Since pytrees can be flattened, that structure is isomorphic to a list of
# lists (with no further nesting).

SamplerState = namedtuple("SamplerState",
                                                    ["packed_state", "tree_def", "subtree_defs"])
state_flatten_fn = lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs))
state_unflatten_fn = lambda data, xs: SamplerState(xs[0], data[0], data[1])
register_pytree_node(SamplerState, state_flatten_fn, state_unflatten_fn)

SamplerKeys = namedtuple("SamplerKeys", ["keys"])
key_flatten_fn = lambda xs: ((xs.keys,), (1,)) #make the (1,) an empty tuple
key_unflatten_fn = lambda data, xs: SamplerKeys(xs[0])
register_pytree_node(SamplerKeys, key_flatten_fn, key_unflatten_fn)

SamplerFns = namedtuple("SamplerFns", 
    ['init', 'propose', 'accept', 'update', 'get_params']
)

def check_equal_pytrees(tree1, tree2, prefix=""):
    if tree1 != tree2: #compares tree defs of the two trees
        msg = (prefix +
               "Passed a gradient tree that did "
               "not match the parameter tree structure with which it was "
               "initialized: parameter tree {} and grad tree {}.")
        raise TypeError(msg.format(tree2, tree1))


def sampler(sampler_builder):
    """Decorator to make an sampler defined for arrays generalize to containers.

    With this decorator, you can write init, propose, update, and get_params functions that
    each operate only on single arrays, and convert (or tree-ify) them to corresponding
    functions that operate on pytrees of parameters. See the samplers defined in
    sampler_fns.py for examples.

    Args:
        sampler_builder: a function that returns an ``(init, propose, update, get_params)``
            triple of functions that might only work with ndarrays, as per

    Returns:
        An ``(init, propose, update, get_params)`` triple of functions that work on
        arbitrary pytrees.
    """
    @functools.wraps(sampler_builder)
    def tree_sampler_builder(*args, **kwargs):
        init, propose, log_proposal, update, get_params, init_key = sampler_builder(*args, **kwargs)

        @functools.wraps(init)
        def tree_init(x0_tree):
            x0_flat, tree = tree_flatten(x0_tree) #tree is the treedef
            initial_keys = jax.random.split(init_key, len(x0_flat))
            initial_states, initial_keys = unzip2(init(x0, k) for x0, k in \
                                                  zip(x0_flat, initial_keys))
            states_flat, subtrees = unzip2(map(tree_flatten, initial_states))
            return SamplerState(states_flat, tree, subtrees), SamplerKeys(initial_keys)

        @functools.wraps(propose)
        # def tree_propose(i, grad_tree, samp_state, samp_keys):
        def tree_propose(i, grad_tree, samp_state, samp_keys, **kwargs):
            states_flat, tree, subtrees = samp_state
            keys = samp_keys
            grad_flat, tree2 = tree_flatten(grad_tree)
            if tree2 != tree: #compares tree defs of the two trees
                msg = ("sampler propose function was passed a gradient tree that did "
                       "not match the parameter tree structure with which it was "
                       "initialized: parameter tree {} and grad tree {}.")
                raise TypeError(msg.format(tree, tree2))
            states = map(tree_unflatten, subtrees, states_flat)
            keys, keys_meta = tree_flatten(keys)

            _propose = lambda g, x, k: propose(i, g, x, k, **kwargs)
            new_states, new_keys = unzip2(map(partial(propose, i), grad_flat, states, keys))
            new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
            for subtree, subtree2 in zip(subtrees, subtrees2):
                if subtree2 != subtree:
                    msg = ("sampler propose function produced an output structure that "
                           "did not match its input structure: input {} and output {}.")
                    raise TypeError(msg.format(subtree, subtree2))

            return SamplerState(new_states_flat, tree, subtrees), SamplerKeys(new_keys)

        def tree_accept(
            i,
            logp,
            logp_prop,
            grad_tree,
            state,
            prop_grad_tree,
            prop_state,
            samp_keys
        ):

            states_flat, tree, subtrees = state
            grad_flat, tree2 = tree_flatten(grad_tree)

            prop_states_flat, prop_tree, prop_subtrees = prop_state
            prop_grad_flat, prop_tree2 = tree_flatten(prop_grad_tree)

            # compute logQ(xprop|x)
            logq_prop = sum(
                map(partial(log_proposal, i), 
                    grad_flat, states_flat, prop_grad_flat, prop_states_flat
                )
            )
            # compute #logQ(x|xprop)
            logq_x = sum(
                map(partial(log_proposal, i), 
                    prop_grad_flat, prop_states_flat, grad_flat, states_flat
                )
            ) 
            
            # compute log_alpha = log(P(xprop)/P(x) * Q(x|xprop)/Q(xprop|x))
            log_alpha = logp_prop + logq_x - logp - logq_prop
            
            # split a single key from the tree keys for the global acceptance step
            samp_keys_flat, samp_keys_meta = tree_flatten(samp_keys)
            global_key, next_key = jax.random.split(samp_keys_flat[0])
            samp_keys_flat[0] = next_key
            
            # perform global acceptance step (sample accept/reject for each tree)
            # not each leaf of each tree
            U = jax.random.uniform(global_key, (logp.shape[0],))
            accept_idxs = jnp.log(U) < log_alpha
            
            return accept_idxs, SamplerKeys(samp_keys_flat)

        @functools.wraps(update)
        def tree_update(
            i, 
            accept_idxs, 
            grad_tree,
            samp_state, 
            prop_grad_tree,
            prop_samp_state,
            samp_keys
        ):
            """
            logp_x and logp_xprop are (N,) arrays (i.e. arrays of scalars)
            of log probs to be passed to every call of update.
            """
            keys = samp_keys

            states_flat, tree, subtrees = samp_state
            grad_flat, tree2 = tree_flatten(grad_tree)
            # compare state grad tree and state tree
            check_equal_pytrees(tree, tree2, prefix='state tree vs grad tree: ')
            
            prop_states_flat, prop_tree, prop_subtrees = prop_samp_state
            prop_grad_flat, prop_tree2 = tree_flatten(prop_grad_tree)
            # compare proposal tree and state tree
            check_equal_pytrees(prop_tree, tree, prefix='prop state tree vs grad tree: ')

            # compare proposal grad tree and state grad tree 
            check_equal_pytrees(prop_tree2, prop_tree, prefix='prop state tree vs prop grad tree: ')            

            states = map(tree_unflatten, subtrees, states_flat)
            prop_states = map(tree_unflatten, prop_subtrees, prop_states_flat)
            keys, keys_meta = tree_flatten(keys)
            new_states, new_keys= unzip2(
                map(partial(update, i, accept_idxs),
                grad_flat, states, prop_grad_flat, prop_states, keys)
            )
            new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
            for subtree, subtree2 in zip(subtrees, subtrees2):
                if subtree2 != subtree:
                    msg = ("sampler update function produced an output structure that "
                           "did not match its input structure: input {} and output {}.")
                    raise TypeError(msg.format(subtree, subtree2))

            return SamplerState(new_states_flat, tree, subtrees), SamplerKeys(new_keys)

        @functools.wraps(get_params)
        def tree_get_params(samp_state, **kwargs):
            states_flat, tree, subtrees = samp_state
            states = map(tree_unflatten, subtrees, states_flat)
            params = (get_params(s, **kwargs) for s in states)
            return tree_unflatten(tree, params)

        return SamplerFns(tree_init, tree_propose, tree_accept, tree_update, tree_get_params)
    return tree_sampler_builder