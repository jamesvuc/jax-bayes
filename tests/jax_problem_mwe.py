
import jax
from jax import grad, vmap
import jax.numpy as np
import jax.scipy.stats.norm as norm

from tqdm import trange

import numpy as onp

def bimodal_logprob(z):
	return np.log(np.sin(z)**2) + np.log(np.sin(2*z)**2) + norm.logpdf(z)




def langevin(logprob, x0, num_iters = 1000, num_samples=10, step_size = 0.01, callback = None):
	"""
	logprob:	autograd.np-valued function that accepts array x of shape x0.shape 
				and returns array of shape (x.shape[0],) representing log P(x)
				up to a constant factor.
		x0: 	inital array of inputs.
	"""
	x = x0
	g = vmap(grad(logprob))

	key = jax.random.PRNGKey(0)
	Zs = jax.random.normal(key, (num_iters, x.shape[0]))
	
	for i in trange(num_iters):
		Z = Zs[i]
		# print(Z.shape)
		# input()
		dx = g(x)
		if callback: callback(x,i,dx)
		x += 0.5*(step_size**2)*dx + step_size*Z

	return x


if __name__=='__main__':
	n_iters, n_samples = 2500, 500
	key = jax.random.PRNGKey(0)
	# init_vals = jax.random.normal(key, (n_samples, 1))
	init_vals = jax.random.normal(key, (n_samples,))
	
	logprob = bimodal_logprob
	
	samps = langevin(logprob, init_vals,
					 num_iters = n_iters, num_samples = n_samples, step_size = 0.05)
	
	print(samps)
