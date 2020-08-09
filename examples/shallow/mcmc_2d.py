import numpy as onp

import jax
import jax.numpy as np
import jax.scipy.stats.multivariate_normal as mvn

import itertools, math
import datetime as dt
from matplotlib import pyplot as plt
import seaborn as sns

import sys; sys.path.append('../..')
from jax_bayes.mcmc import (langevin_fns, mala_fns, rk_langevin_fns,
							hmc_fns, rwmh_fns, rms_langevin_fns)
from jax_bayes.mcmc import bb_mcmc

def make_logprob():

	mus = np.array([[0.0, 0.0],
					[2.5,  4.0],
					[4.0,  0.0]])

	sigmas = np.array([	[[1.0, 0.0],
					 	 [0.0, 2.0]],
						
						[[2.0, -1.0],
					  	 [-1.0, 1.0]],

						[[1.0, 0.1],
						 [0.1, 2.0]] ])

	@jax.jit
	def _logprob(z):
		return np.log(mvn.pdf(z, mean=mus[0], cov=sigmas[0]) + \
			   		  mvn.pdf(z, mean=mus[1], cov=sigmas[1]) + \
			   		  mvn.pdf(z, mean=mus[2], cov=sigmas[2]))

	return _logprob


def main():
	#====== Setup =======
	n_iters, n_samples, d = 2000, 2000, 2
	key = jax.random.PRNGKey(1)
	init_vals = np.array([2.0, 2.0])


	logprob = make_logprob()
	allsamps = []

	#====== Tests =======

	t = dt.datetime.now()
	print('running 2d tests ...')
	samps = bb_mcmc(logprob, init_vals, langevin_fns, num_iters = n_iters, 
				num_samples = n_samples, seed=0, step_size = 0.05, 
				init_dist='uniform', init_stddev=5.0)
	print('done langevin in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)



	t = dt.datetime.now()
	samps = bb_mcmc(logprob, init_vals, mala_fns, num_iters = n_iters, 
				num_samples = n_samples, seed=0, step_size = 0.05, 
				init_dist='uniform', init_stddev=5.0, recompute_grad=True)
	print('done MALA in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	
	t = dt.datetime.now()
	samps = bb_mcmc(logprob, init_vals, rk_langevin_fns, num_iters = n_iters, 
				num_samples = n_samples, seed=0, step_size = 0.05, 
				init_dist='uniform', init_stddev=5.0, recompute_grad=True)
	print('done langevin_RK in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	t = dt.datetime.now()
	samps = bb_mcmc(logprob, init_vals, hmc_fns, num_iters = n_iters//5, 
				proposal_iters = 5, num_samples = n_samples, seed=0, step_size = 0.05, 
				init_dist='uniform', init_stddev=5.0, recompute_grad=True)
	print('done HMC in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	t = dt.datetime.now()
	samps = bb_mcmc(logprob, init_vals, rms_langevin_fns, num_iters = n_iters, 
				num_samples = n_samples, seed=0, step_size = 1e-3, #1e-3 
				beta=0.99, eps=1e-5,
				init_dist='uniform', init_stddev=5.0)
	print('done rms_langevin in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)


	t = dt.datetime.now()
	samps = bb_mcmc(logprob, init_vals, rwmh_fns, num_iters = n_iters, 
				num_samples = n_samples, seed=0, step_size = 0.05, 
				init_dist='uniform', init_stddev=5.0)
	print('done RW MH in' , dt.datetime.now()-t,'\n')
	allsamps.append(samps)


	#====== Plotting =======
	init_vals = jax.random.uniform(key, (n_samples,d)) *  5.0

	pts = onp.linspace(-7, 7, 1000)
	X, Y = onp.meshgrid(pts, pts)
	pos = onp.empty(X.shape + (2,))
	pos[:, :, 0] = X
	pos[:, :, 1] = Y
	Z = onp.exp(jax.vmap(logprob)(pos))

	""" for the solo contour plot 
	f, ax = plt.subplots()
	ax.contour(X, Y, Z,)
	ax.set_xlim(-2, 6)
	ax.set_ylim(-5, 7)
	plt.show()
	"""

	names = ['langevin', 'MALA', 'langevin_RK', 'HMC', 'RMS_langevin', 'RWMH']
	
	cols = 2
	rows = math.ceil(len(names) / cols)
	idxs = itertools.product(range(rows), range(cols))
	f, axes = plt.subplots(rows, cols, sharex=True, figsize=(12, 9))
	
	for i, (name, samps, (r,c)) in enumerate(zip(names, allsamps, idxs)):
		print(name)
		row = i // 3
		col = i % 3
		ax = axes[r, c]

		ax.contour(X, Y, Z, alpha=0.5, cmap='Oranges')
		# ax.hist2d(samps[:,0], samps[:,1], alpha=0.5, bins=25)
		# sns.jointplot(samps[:,0], samps[:,1], kind="kde", height=7, space=0, ax=ax)
		sns.kdeplot(samps[:,0], samps[:,1], shade=True, shade_lowest=True,
					cmap='Blues', ax=ax)
		ax.set_title(name)
		ax.set_xlim(-2, 6)
		ax.set_ylim(-5, 7)
	

	# axes[1,2].hist2d(init_vals[:,0], init_vals[:,1], bins=25)
	# axes[1,2].set_title('Example Initial samples.')

	plt.show()




if __name__ == '__main__':
	main()

	



