import numpy as onp
import jax.numpy as jnp
import jax.scipy.stats.norm as norm
import jax.scipy.stats.multivariate_normal as mvn
from jax import vmap, jit
import jax
from jax.experimental import optimizers

from copy import copy, deepcopy
import datetime as dt
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

import sys; sys.path.append('../..')
from jax_bayes.variational import diagonal_mvn_fns, elbo_reparam
# from jax_bayes.variational.bbvi import elbo_reparam, elbo_noscore, bbvi
# from jax_bayes.variational.vf import elbo_reparam, elbo_noscore

@jit
@vmap
def logprob(z):
	x, y = z[0], z[1]
	y_density = norm.logpdf(y, 0, 1.35)
	x_density = norm.logpdf(x, 0, jnp.exp(y))

	return x_density + y_density

def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
	x = jnp.linspace(*xlimits, num=numticks)
	y = jnp.linspace(*ylimits, num=numticks)
	X, Y = jnp.meshgrid(x, y)
	zs = func(jnp.concatenate([jnp.atleast_2d(X.ravel()), 
							  jnp.atleast_2d(Y.ravel())]).T)
	Z = zs.reshape(X.shape)
	Z = jax.lax.stop_gradient(Z)
	ax.contour(X, Y, Z)
	ax.set_yticks([])
	ax.set_xticks([])

def main():
	x0 = jnp.zeros(2)

	# var_approx = DiagonalMVN(init_mean, init_logcov)
	# key = jax.random.PRNGKey(0)
	key = jax.random.PRNGKey(1)
	vf = diagonal_mvn_fns(key, mean_stddev=0.1)
	var_params, var_keys = vf.init(x0)

	print('init')

	lr = 1e-3
	# lr = 1e-2
	opt_init, opt_update, opt_get_params = optimizers.adam(lr)
	opt_state = opt_init(var_params)

	f, axes = plt.subplots(2, figsize=(7, 7))
	f.subplots_adjust(bottom=0.05, top=0.95, hspace=0.3)
	pts, hist = [], []
	def callback(i, fx, params,  *args):
		#using the fact that we're updating var approx in-place
		pts.append(i)
		hist.append(-fx)
		ax = axes[0]
		ax.cla()
		ax.plot(pts, hist)

		ax = axes[1]
		ax.cla()
		ax.set_title(f"i={i}")
		plot_isocontours(ax, lambda z:jnp.exp(logprob(z)))

		mean, logcov = params
		cov = cov=jnp.diag(jnp.exp(logcov)) + 1e-6
		logq = lambda z:jnp.exp(mvn.logpdf(z, mean, cov))
		logq = jax.vmap(logq)
		plot_isocontours(ax,logq)
		plt.pause(1.0/30)
		plt.draw()

	num_samples = 1000
	
	# def obj(p):
	def elbo(p, keys):
		samples_state, _ = vf.sample(0, num_samples, keys, p)
		samples = vf.get_samples(samples_state)
		# return jnp.mean(logprob(samples) + vf.evaluate(samples_state, p))
		return jnp.mean(logprob(samples)) + vf.entropy(p)

	print('elbo before:')
	params = opt_get_params(opt_state)
	_elbo = elbo(params, var_keys)
	print(_elbo)

	@jax.jit
	def bbvi_step(i, opt_state, var_keys):
		params = opt_get_params(opt_state)
		var_keys = vf.next_key(var_keys) #generate one key to use now
		next_keys = vf.next_key(var_keys) #generate one to return
		
		obj = lambda p: - elbo(p, var_keys)

		# dlambda = jax.grad(obj)(params)
		loss, dlambda = jax.value_and_grad(obj)(params)
		opt_state = opt_update(i, dlambda, opt_state)
		return opt_state, next_keys, loss

	# for i in trange(1000):
	for i in trange(5000):
		opt_state, var_keys, loss = bbvi_step(i, opt_state, var_keys)
		if i % 10 == 0:
			callback(i, loss, vf.get_params(opt_get_params(opt_state)))

	print('done')
	# print(opt_get_params(opt_state))
	params = opt_get_params(opt_state)
	_elbo = elbo(params, var_keys)
	print('elbo after', _elbo)
	plt.show()

	# elbo = elbo_reparam(logprob, var_approx, num_samples=1000)
	# print('elbo after=', elbo)

	

if __name__ == '__main__':
	main()