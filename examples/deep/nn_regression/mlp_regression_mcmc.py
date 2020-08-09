import haiku as hk

import jax.numpy as jnp
from jax.experimental import optimizers
import jax

from tqdm import tqdm, trange

import copy
import numpy as np

import sys; sys.path.append('../../..')
from jax_bayes.mcmc import langevin_fns, mala_fns, rk_langevin_fns, hmc_fns
from jax_bayes.utils import confidence_bands

from matplotlib import pyplot as plt
from copy import copy

def build_dataset():
	np.random.seed(0)
	n_train, n_test, d = 200, 100, 1
	xlims = [-1.0, 5.0]
	x_train = np.random.rand(n_train, d) * (xlims[1] - xlims[0]) + xlims[0]
	x_test = np.random.rand(n_test, d) * (xlims[1] - xlims[0]) + xlims[0]

	target_func = lambda t: (np.log(t + 100.0) * np.sin(1.0 * np.pi*t)) + 0.1 * t

	y_train = target_func(x_train)
	y_test = target_func(x_test)

	y_train += np.random.randn(*x_train.shape) * (1.0 * (x_train + 2.0)**0.5)


	return (x_train, y_train), (x_test, y_test)

def net_fn(x):

	# sig = 5.0 #for deterministic setup
	""" stddev(X + Y) = sqrt(stddev(X)**2 + stddev(Y)**2)
		therefore if you want wgts w/ stddev = 5,
		and init sigma 4, you need random perturbation sigma = 3
		bc 3**2 + 4**2 = 5**2 """
	sig = 4.0
	mlp = hk.Sequential([
		hk.Linear(128, w_init=hk.initializers.RandomNormal(stddev=sig), 
					   b_init=hk.initializers.RandomNormal(stddev=sig)), 
		jnp.tanh, 
		hk.Linear(1,   w_init=hk.initializers.RandomNormal(stddev=sig), 
					   b_init=hk.initializers.RandomNormal(stddev=sig))
		])

	return mlp(x)

def main():
	# ======= Setup =======
	xy_train, xy_test = build_dataset()
	(x_train, y_train), (x_test, y_test) = xy_train, xy_test

	# lr = 1e-3
	# lr = 1e-2 #for langevin and mala
	lr = 1e-3 #for rk_langevin
	reg = 0.1
	# reg = 0.0
	# lik_var = 0.1
	lik_var = 0.5

	#hk.transform returns a Transformed object with methods init and apply
	net = hk.transform(net_fn)
	key = jax.random.PRNGKey(0)
	# sampler_init, sampler_propose, sampler_update, sampler_get_params = \
	# 	langevin_fns(key, num_samples=10, step_size=lr, init_stddev=3.0)
	# sampler_init, sampler_propose, sampler_update, sampler_get_params = \
	# 	mala_fns(key, num_samples=10, step_size=lr, init_stddev=3.0)
	# sampler_init, sampler_propose, sampler_update, sampler_get_params = \
	# 	rk_langevin_fns(key, num_samples=10, step_size=lr, init_stddev=3.0)
	sampler_init, sampler_propose, sampler_update, sampler_get_params = \
		hmc_fns(key, num_samples=10, step_size=lr, init_stddev=3.0)



	def logprob(params, xy):
		""" log posterior, assuming 
			P(params) ~ N(0,eta)
			P(y|x, params) ~ N(f(x;params), lik_var)
		"""
		x, y = xy
		
		preds = net.apply(params, None, x)
		log_prior = - reg * sum(jnp.sum(jnp.square(p)) 
							for p in jax.tree_leaves(params))
		log_lik = - jnp.mean(jnp.square(preds - y)) / lik_var
		return log_lik + log_prior

	@jax.jit
	def sampler_step(i, sampler_state, keys, batch):
		params = sampler_get_params(sampler_state)
		logp = lambda params:logprob(params, batch) #can make this 1-line?
		# dx = jax.vmap(jax.grad(logp))(params)
		fx, dx = jax.vmap(jax.value_and_grad(logp))(params)

		# sampler_prop_state, new_keys = sampler_propose(i, dx, sampler_state, keys)

		# # fx_prop, dx_prop = fx, dx
		# prop_params = sampler_get_params(sampler_prop_state)
		# fx_prop, dx_prop = jax.vmap(jax.value_and_grad(logp))(prop_params)

		sampler_prop_state, dx_prop, new_keys = sampler_state, dx, keys
		for j in range(5): #5 iterations of the leapfrog integrator
			sampler_prop_state, new_keys = \
				sampler_propose(i, dx_prop, sampler_prop_state, new_keys)
			prop_params = sampler_get_params(sampler_prop_state)
			fx_prop, dx_prop = jax.vmap(jax.value_and_grad(logp))(prop_params)


		sampler_state, new_keys = sampler_update(i, fx, fx_prop, 
												 dx, sampler_state, 
												 dx_prop, sampler_prop_state, 
												 new_keys)
		
		return sampler_state, new_keys

	params = net.init(jax.random.PRNGKey(42), x_train)
	sampler_state, sampler_keys = sampler_init(params)
	
	# for step in trange(5):
	for step in trange(5000):
		if step % 250 == 0:
			sampler_params = sampler_get_params(sampler_state)
			logp = lambda params:logprob(params, xy_train)
			train_logp = jnp.mean(jax.vmap(logp)(sampler_params))
			logp = lambda params:logprob(params, xy_test )
			test_logp = jnp.mean(jax.vmap(logp)(sampler_params))
			print(f"step = {step}"
				  f" | train logp = {train_logp:.3f}"
				  f" | test logp = {test_logp:.3f}")
		
		sampler_state, sampler_keys = \
			sampler_step(step, sampler_state, sampler_keys, xy_train)


	sampler_params = sampler_get_params(sampler_state)

	# ===========
	# plot_inputs = np.linspace(-1, 6, num=400).reshape(-1,1)
	plot_inputs = np.linspace(-1, 10, num=600).reshape(-1,1)
	outputs = jax.vmap(lambda params: net.apply(params, None, plot_inputs))(sampler_params)

	# lower, upper = gaussian_conf_bands(outputs.squeeze(-1).T)
	lower, upper = confidence_bands(outputs.squeeze(-1).T)
	
	f, axes = plt.subplots(1)
	
	ax = axes
	ax.plot(x_train.ravel(), y_train.ravel(), 'bx', color='green')
	ax.plot(x_test.ravel(), y_test.ravel(), 'bx', color='red')
	for i in range(outputs.shape[0]):
		ax.plot(plot_inputs, outputs[i], alpha=0.25)
	ax.plot(plot_inputs, np.mean(outputs[:10, :, 0].T, axis=1), color='black', linewidth=1.0)
	
	ax.fill_between(plot_inputs.squeeze(-1), lower, upper, alpha=0.75)

	ax.set_ylim(-10, 15)

	plt.show()

if __name__ == '__main__':
	main()