import numpy as np
np.random.seed(0)

import haiku as hk

import jax.numpy as jnp
from jax.experimental import optimizers
import jax

from tqdm import tqdm, trange
from matplotlib import pyplot as plt

def build_dataset():
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
	xy_train, xy_test = build_dataset()
	(x_train, y_train), (x_test, y_test) = xy_train, xy_test

	lr = 1e-3
	reg = 0.0
	lik_var = 0.5

	net = hk.transform(net_fn)
	opt_init, opt_update, opt_get_params = optimizers.sgd(lr)

	def logprob(params, xy):
		""" log posterior logP(params | xy), assuming 
		P(params) ~ N(0,eta)
		P(y|x, params) ~ N(f(x;params), lik_var)
		"""
		x, y = xy
		preds = net.apply(params, None, x)
		log_prior = - reg * sum(jnp.sum(jnp.square(p)) 
							for p in jax.tree_leaves(params))
		log_lik = - jnp.mean(jnp.square(preds - y)) / lik_var
		return log_lik + log_prior

	#minimize the - logprob to find MAP
	loss = lambda params, xy: - logprob(params, xy)

	@jax.jit
	def train_step(i, opt_state, batch):
		params = opt_get_params(opt_state)
		dx = jax.grad(loss)(params, batch)
		opt_state = opt_update(i, dx, opt_state)
		return opt_state

	# ======= Training ======

	# initialization
	params = net.init(jax.random.PRNGKey(42), x_train)
	opt_state = opt_init(params)

	#do the optimization
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
	
	# ========= Plotting =========
	plot_inputs = np.linspace(-1, 10, num=600).reshape(-1,1)
	outputs = net.apply(params, None, plot_inputs)

	f, ax = plt.subplots(1)

	ax.plot(x_train.ravel(), y_train.ravel(), 'bx', color='green')
	ax.plot(x_test.ravel(), y_test.ravel(), 'bx', color='red')
	ax.plot(plot_inputs, outputs, alpha=1)
	
	plt.show()

if __name__ == '__main__':
	main()