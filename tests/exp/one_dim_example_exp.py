#1d_tests.py

# import autograd.numpy as np
# import autograd.scipy.stats.norm as norm
import numpy as onp
import jax.numpy as np
import jax.scipy.stats.norm as norm
from jax import vmap, jit
import jax

from copy import copy, deepcopy
import datetime as dt
from matplotlib import pyplot as plt
import seaborn as sns

import sys; sys.path.append('../..')
# from jax_bayes.mcmc.langevin2 import langevin2 as langevin
# from jax_bayes.mcmc.langevin3 import langevin2 as langevin
from jax_bayes.mcmc.langevin3 import (langevin_fns,
									  mala_fns,
									  rk_langevin_fns,
									  hmc_fns)
from jax_bayes.mcmc.utils import bb_mcmc
# # from jax_bayes.mcmc.langevin import langevin, MALA, RK_langevin
# from jax_bayes.mcmc.langevin_old import MALA, RK_langevin
# from jax_bayes.mcmc.hamiltonian import HMC
# from jax_bayes.mcmc.metropolis import RWMH
from new_sampler import rms_langevin_fns

@jit 
def bimodal_logprob(z):
	# return np.log(np.sin(z)**2) + np.log(np.sin(2*z)**2) + norm.logpdf(z)
	return np.log(np.sin(z)**2) + np.log(np.sin(2*z)**2) + norm.logpdf(z)

def main():

	#====== Setup =======

	n_iters, n_samples = 1000, 500 #main
	# n_iters, n_samples = 10, 5
	# n_iters, n_samples = 10, 500
	key = jax.random.PRNGKey(0)
	# init_vals = jax.random.normal(key, (n_samples,))
	# init_vals = np.array([0.0])
	init_vals = np.array(0.0)

	allsamps = []
	logprob = bimodal_logprob
	# logprob = vmap(bimodal_logprob)

	#====== Tests =======

	t = dt.datetime.now()
	print('running 1d tests ...')
	samps = bb_mcmc(logprob, init_vals, langevin_fns, num_iters = n_iters, 
					 num_samples = n_samples, step_size = 0.05, 
					 init_dist='normal', init_stddev=1.0)
	print('done langevin in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	t = dt.datetime.now()
	samps = bb_mcmc(logprob, init_vals, mala_fns, num_iters = n_iters, 
					 num_samples = n_samples, step_size = 0.05, 
					 init_dist='normal', init_stddev=1.0, recompute_grad=True)
	print('done MALA in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)
	

	t = dt.datetime.now()
	samps = bb_mcmc(logprob, init_vals, rk_langevin_fns, num_iters = n_iters, 
					 num_samples = n_samples, step_size = 0.01, 
					 init_dist='normal', init_stddev=1.0, recompute_grad=True)
	print('done langevin_RK in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)


	t = dt.datetime.now()
	samps = bb_mcmc(logprob, init_vals, hmc_fns, num_iters = n_iters//10, 
					 proposal_iters= 5, num_samples = n_samples, step_size = 0.05, 
					 init_dist='normal', init_stddev=1.0, recompute_grad=True)
	print('done HMC in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)
	
	# ---------------
	# init_vals = jax.random.normal(key, (n_samples,))

	# t = dt.datetime.now()
	# samps = RWMH(logprob, copy(init_vals), 
	# 			  num_iters = n_iters, num_samples = n_samples, sigma = 0.5)
	# print('done RW MH in' , dt.datetime.now()-t,'\n')

	t = dt.datetime.now()
	# """
	samps =  bb_mcmc(logprob, init_vals, rms_langevin_fns, 
					num_iters = n_iters, num_samples = n_samples,
					step_size = 0.05, beta=0.9,
					init_dist='normal',  init_stddev=1.0)
	# """
	"""
	samps =  bb_mcmc(logprob, init_vals, rms_langevin_fns, 
					num_iters = n_iters, num_samples = 5, use_jit=False,
					step_size = 0.05, beta=0.9,
					init_dist='normal',  init_stddev=1.0)
	# """
	print('done new in' , dt.datetime.now()-t,'\n')
	allsamps.append(samps)


	# t = dt.datetime.now()
	# samps = HMC(logprob, copy(init_vals),
	# 			num_iters = n_iters//5, num_samples = n_samples, 
	# 			step_size = 0.05, num_leap_iters=5)
	# print('done HMC in', dt.datetime.now()-t,'\n')
	# allsamps.append(samps)

	#====== Plotting =======

	lims = [-5,5]
	# names = ['langevin', 'MALA', 'langevin_RK', 'RW MH', 'HMC']
	names = ['langevin', 'MALA', 'langevin_RK', 'HMC', 'RWMH']
	# names = ['langevin', 'MALA', 'langevin_RK', 'HMC']
	f, axes = plt.subplots(len(names), sharex=True)
	if len(names) == 1:
		axes = [axes]
	for i, (name, samps) in enumerate(zip(names, allsamps)):

		sns.distplot(samps, bins=1000, kde=False, ax=axes[i])
		axb = axes[i].twinx()
		axb.scatter(samps, np.ones(len(samps)), alpha=0.1, marker='x', color='red')
		axb.set_yticks([])

		zs = np.linspace(*lims, num=250)
		axes[i].twinx().plot(zs, np.exp(bimodal_logprob(zs)), color='orange')

		axes[i].set_xlim(*lims)
		title = name
		axes[i].set_title(title)

	plt.show()


if __name__ == '__main__':
	main()