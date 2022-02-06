#mcmc_1d.py
import jax.numpy as jnp
import jax.scipy.stats.norm as norm
import jax

from copy import copy, deepcopy
import itertools, math
import time
from matplotlib import pyplot as plt
import seaborn as sns

from jax_bayes.mcmc import (
	langevin_fns,
	mala_fns,
	rk_langevin_fns,
	hmc_fns,
	rms_langevin_fns,
	rwmh_fns
)
from jax_bayes.mcmc import blackbox_mcmc as bb_mcmc

@jax.jit 
def bimodal_logprob(z):
	return jnp.log(jnp.sin(z)**2) + jnp.log(jnp.sin(2*z)**2) + norm.logpdf(z)

def main():

	#====== Setup =======

	n_iters, n_samples = 1000, 1000
	seed = 0
	init_vals = jnp.array(0.0)

	allsamps = []
	logprob = bimodal_logprob
	
	#====== Tests =======

	t = time.time()
	print('running 1d tests ...')
	samps = bb_mcmc(
		logprob, init_vals, langevin_fns, num_iters=n_iters, 
		seed=seed, num_samples=n_samples, step_size=1e-3,
		init_dist='normal', init_stddev=1.0
	)
	print('done langevin in', time.time()-t,'\n')
	allsamps.append(samps)

	t = time.time()
	samps = bb_mcmc(
		logprob, init_vals, mala_fns, num_iters=n_iters, 
		seed=seed, num_samples=n_samples, step_size=1e-3,
		init_dist='normal', init_stddev=1.0, recompute_grad=True
	)
	print('done MALA in', time.time()-t,'\n')
	allsamps.append(samps)
	
	t = time.time()
	samps = bb_mcmc(
		logprob, init_vals, rk_langevin_fns, num_iters=n_iters, 
		seed=seed, num_samples=n_samples, step_size=1e-3, 
		init_dist='normal', init_stddev=1.0, recompute_grad=True
	)
	print('done langevin_RK in', time.time()-t,'\n')
	allsamps.append(samps)


	t = time.time()
	samps = bb_mcmc(
		logprob, init_vals, hmc_fns, num_iters=n_iters, 
		proposal_iters= 5, seed=seed, num_samples=n_samples, step_size=1e-2, 
		init_dist='normal', init_stddev=1.0, recompute_grad=True
	)
	print('done HMC in', time.time()-t,'\n')
	allsamps.append(samps)

	t = time.time()
	samps = bb_mcmc(
		logprob, init_vals, rms_langevin_fns, num_iters=n_iters, 
		seed = seed, num_samples = n_samples, step_size=5e-3, 
		init_dist='normal', init_stddev=1.0, beta=0.99
	)
	print('done rms in', time.time()-t,'\n')
	allsamps.append(samps)

	t = time.time()
	samps = bb_mcmc(
		logprob, init_vals, rwmh_fns, num_iters=n_iters, 
		seed=seed, num_samples=n_samples, step_size=0.05, 
		init_dist='normal', init_stddev=1.0, recompute_grad=True,
	)
	print('done rwmh in', time.time()-t,'\n')
	allsamps.append(samps)

	#====== Plotting =======

	lims = [-5,5]
	names = [
		'langevin',
		'MALA',
		'langevin_RK',
		'HMC',
		'RMS langevin',
		'RWMH'
	]
	cols = 2
	rows = math.ceil(len(names) / cols)
	idxs = itertools.product(range(rows), range(cols))
	f, axes = plt.subplots(rows, cols, sharex=True, figsize=(12, 8))
	for i, (name, samps, (r,c)) in enumerate(zip(names, allsamps, idxs)):
		print(samps.shape)
		sns.distplot(samps, bins=500, kde=False, ax=axes[r,c])
		axb = axes[r,c].twinx()
		axb.scatter(samps, jnp.ones(len(samps)), alpha=0.1, marker='x', color='red')

		zs = jnp.linspace(*lims, num=250)
		axc = axes[r,c].twinx()
		axc.plot(zs, jnp.exp(bimodal_logprob(zs)), color='orange')

		axes[r,c].set_xlim(*lims)
		title = name
		axes[r,c].set_title(title)

		axes[r,c].set_yticks([])
		axb.set_yticks([])
		axc.set_yticks([])

	plt.show()


if __name__ == '__main__':
	main()