import haiku as hk

import jax.numpy as jnp
from jax.experimental import optimizers
import jax

from tqdm import tqdm, trange



import numpy as np
np.random.seed(0)

# import autograd.numpy as np

# import sys; sys.path.append('..')
# from mcmc import langevin, RWMH, HMC, RK_langevin

from matplotlib import pyplot as plt
from copy import copy

def build_dataset():
	n_train, n_test, d = 200, 100, 1
	xlims = [-1.0, 5.0]
	x_train = np.random.rand(n_train, d) * (xlims[1] - xlims[0]) + xlims[0]
	x_test = np.random.rand(n_test, d) * (xlims[1] - xlims[0]) + xlims[0]
	# x_train = np.random.rand(n_train) * (xlims[1] - xlims[0]) + xlims[0]
	# x_test = np.random.rand(n_test) * (xlims[1] - xlims[0]) + xlims[0]

	target_func = lambda t: (np.log(t + 100.0) * np.sin(1.0 * np.pi*t)) + 0.1 * t

	y_train = target_func(x_train)
	y_test = target_func(x_test)

	y_train += np.random.randn(*x_train.shape) * (1.0 * (x_train + 2.0)**0.5)


	return (x_train, y_train), (x_test, y_test)

def gaussian_conf_bands(data, alpha=0.68):
	"""
	Y columns are independent samples, 
	(X[i], Y[i]) is a set of Y.shape[1] samples at X[i])
	"""
	X,Y = data
	U,L = [],[]
	for y in Y:
		mu = np.mean(y)
		sigma = np.std(y) #no dof correction...
		L.append(mu - sigma)
		U.append(mu + sigma)

	return X, L, U

def net_fn(x):

	mlp = hk.Sequential([
		hk.Linear(128, w_init=hk.initializers.RandomNormal(stddev=5.0), 
					   b_init=hk.initializers.RandomNormal(stddev=5.0)), 
		jnp.tanh, 
		hk.Linear(1,   w_init=hk.initializers.RandomNormal(stddev=5.0), 
					   b_init=hk.initializers.RandomNormal(stddev=5.0))
		])

	return mlp(x)

def main():
	# ======= Setup =======

	# rbf = lambda x: np.exp(-x**2)#deep basis function model
	# num_weights, predictions, logprob = \
	# 	make_nn_funs(layer_sizes=[1, 128, 1], L2_reg=0.1,
	# 				noise_variance=0.5, nonlinearity=np.tanh)

	# (x_train, y_train), (x_test, y_test) = build_dataset()
	xy_train, xy_test = build_dataset()
	(x_train, y_train), (x_test, y_test) = xy_train, xy_test


	# lr = 1e-3
	lr = 1e-3
	# reg = 0.1
	reg = 0.0
	# lik_var = 0.5
	# lik_var = 0.1
	lik_var = 0.5

	#hk.transform returns a Transformed object with methods init and apply
	net = hk.transform(net_fn)
	opt_init, opt_update, opt_get_params = optimizers.sgd(lr)
	# opt_init, opt_update, opt_get_params = optimizers.adam(lr)
	# opt = optix.adam(lr)

	def logprob(params, xy):
		""" log posterior logP(params | xy), assuming 
			P(params) ~ N(0,eta)
			P(y|x, params) ~ N(f(x;params), lik_var)
		"""
		x, y = xy
		preds = net.apply(params, x)
		log_prior = - reg * sum(jnp.sum(jnp.square(p)) 
							for p in jax.tree_leaves(params))
		log_lik = - jnp.mean(jnp.square(preds - y)) / lik_var
		return log_lik + log_prior

	# log_posterior = lambda weights: logprob(weights, x_train, y_train)

	# sig = 5.0
	# init_weights = np.random.randn(args.samples, num_weights) * sig

	#minimize the - logprob to find MAP
	loss = lambda params, xy: - logprob(params, xy)

	@jax.jit
	def train_step(i, opt_state, batch):
		params = opt_get_params(opt_state)
		dx = jax.grad(loss)(params, batch)
		opt_state = opt_update(i, dx, opt_state)
		return opt_state


	# ======= Sampling =======

	# weight_samps = langevin(log_posterior, init_weights,  	
	# 						num_iters = args.iters, num_samples = args.samples, 
	# 						step_size = args.step_size, callback = callback)
	
	params = net.init(jax.random.PRNGKey(42), x_train)

	keys = jax.random.split(jax.random.PRNGKey(1),2)
	# for i,(name,p) in enumerate(params.items()):
	# 	print(name, {p:pval.shape for p,pval in p.items()})
	# 	# input()
	# 	p['b'] = jax.random.normal(keys[i], p['b'].shape)
	
	opt_state = opt_init(params)
	for step in trange(2000):
		if step % 200 == 0:
			params = opt_get_params(opt_state)
			train_loss = loss(params, xy_train)
			test_acc = loss(params, xy_test)
			print(f"step = {step}"
				  f" | train loss = {train_loss:.3f}"
				  f" | test loss = {test_acc:.3f}")
		
		opt_state = train_step(step, opt_state, xy_train)

	params = opt_get_params(opt_state)
	# ===========
	# plot_inputs = np.linspace(-1, 6, num=400).reshape(-1,1)
	plot_inputs = np.linspace(-1, 10, num=600).reshape(-1,1)
	# outputs = predictions(weight_samps, np.expand_dims(plot_inputs, 1))\
	# y_pred = net.apply(params, x_test)
	outputs = net.apply(params, plot_inputs)

	# _, lower, upper = gaussian_conf_bands((plot_inputs, outputs[:,:,0].T))

	f, axes = plt.subplots(2)

	# hist = np.array(hist)
	# axes[0].plot(hist, alpha=0.5)
	
	axes[1].plot(x_train.ravel(), y_train.ravel(), 'bx', color='green')
	axes[1].plot(x_test.ravel(), y_test.ravel(), 'bx', color='red')
	axes[1].plot(plot_inputs, outputs, alpha=1)
	# axes[1].plot(x_test, y_pred, alpha=0.25)
	# axes[1].scatter(x_test, y_pred, alpha=1)
	# axes[1].plot(plot_inputs, np.mean(outputs[:10, :, 0].T, axis=1), color='black', linewidth=1.0)
	# axes[1].fill_between(plot_inputs, lower, upper, alpha=0.75)


	plt.show()

if __name__ == '__main__':
	main()