import jax.numpy as jnp
from jax import grad, vmap
import jax

import math

import sys; sys.path.append('../..')
from jax_bayes.mcmc.sampler import sampler

#add to utils.py
centered_uniform = \
	lambda *args, **kwargs:jax.random.uniform(*args, **kwargs) - 0.5
init_distributions = dict(normal=jax.random.normal, 
						  uniform=centered_uniform)

def match_dims(src, target, start_dim=1):
	""" ensures that src has the same number of dims as target
		by padding with empty dimensions *after* start_dim"""
	new_dims = tuple(range(start_dim, len(target.shape)))
	return jnp.expand_dims(src, new_dims)

@sampler
def rms_langevin_fns(base_key, num_samples=10, step_size=1e-3, beta=0.9,
					eps = 1e-9, init_stddev=0.1, init_dist = 'normal'):
	if type(init_dist) is str:
		init_dist = init_distributions[init_dist]

	def init(x0, key):
		init_key, next_key = jax.random.split(key)
		if num_samples == -1: 
			r = jnp.zeros_like(x0)
			return (x0, r), next_key
		X = init_dist(init_key, (num_samples,) + x0.shape)
		r = jnp.zeros_like(X)
		return (x0 + init_stddev * X, r), next_key

	def propose(i, g, x, key):
		key, next_key = jax.random.split(key)
		x, r = x
		Z = jax.random.normal(key, x.shape)

		r = beta * r + (1. - beta) * jnp.square(g)

		# print(g)
		# print(r)
		# input()

		#original
		# xprop = x + 0.5 * (step_size ** 2) * g + step_size * Z

		#should be equivalent to original
		#this might lead to a more natural transition 
		# from deterministic -> mcmc
		# xprop = x + 0.5 * step_size * g + math.sqrt(step_size) * Z
		
		#adaptive step size = step_size / sqrt(r)
		# xprop = x + 0.5 * ((step_size ** 2) / (r + eps)) * g \
		# 				+ (step_size / (jnp.sqrt(r) + eps)) * Z

		# xprop = x + 0.5 * (step_size / (jnp.sqrt(r) + eps)) * g \
		# 				+ (math.sqrt(step_size) / (jnp.power(r, 0.25) + eps)) * Z

		ss = step_size / (jnp.sqrt(r) + eps)
		xprop = x + 0.5 * ss * g + jnp.sqrt(ss) * Z
		

		return (xprop, r) , next_key

	def update(i, logp_x, logp_xprop, g, x, gprop, xprop, key):
		key, next_key = jax.random.split(key)
		#always accept in ULA
		return xprop, next_key

	def get_params(x):
		x, _ = x
		return x

	return init, propose, update, get_params, base_key

