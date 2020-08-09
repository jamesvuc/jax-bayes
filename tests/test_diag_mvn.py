import jax.numpy as jnp
from jax.experimental import optimizers
import jax

from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.linalg import cholesky
import numpy as np

def diag_mvn_logpdf(x, mean, diag_cov):
	n = mean.shape[-1]
	y = x - mean
	tmp = jnp.einsum('...i,i->...i', y, 1./diag_cov)
	return (-1/2 * jnp.einsum('...i,...i->...',y, tmp) 
		- n/2 * np.log(2*np.pi) - jnp.log(diag_cov).sum()/2.)

#from jax.scipy
def logpdf(x, mean, cov):
	
	from jax.scipy.linalg import solve_triangular as triangular_solve
	if not mean.shape:
		return (-1/2 * jnp.square(x - mean) / cov
				- 1/2 * (np.log(2*np.pi) + jnp.log(cov)))
	else:
		n = mean.shape[-1]
		if not np.shape(cov):
			y = x - mean
			return (-1/2 * jnp.einsum('...i,...i->...', y, y) / cov
			- n/2 * (np.log(2*np.pi) + jnp.log(cov)))
		else:
			if cov.ndim < 2 or cov.shape[-2:] != (n, n):
				raise ValueError("multivariate_normal.logpdf got incompatible shapes")
			L = cholesky(cov)
			y = triangular_solve(L, x - mean, lower=True, transpose_a=True)
			return (-1/2 * jnp.einsum('...i,...i->...', y, y) - n/2*np.log(2*np.pi)
				- jnp.log(L.diagonal()).sum())

if __name__ == '__main__':
	
	d = 3
	mean = jnp.array([0.0, 0.0, 0.0])
	# cov = jnp.array([1.0, 1.0, 1.0])
	cov = jnp.array([1.0, 2.0, 3.0])

	X = jnp.zeros((1, d))

	norm_const = lambda C: - d/2 * (jnp.log(2*jnp.pi) + jnp.log(jnp.prod(C)))
	def norm_const_2(C):
		L = cholesky(C)
		return - d/2*np.log(2*np.pi) - jnp.log(L.diagonal()).sum()

	def new_norm_const(C):
		# print(jnp.log(C))
		# input()
		# return - d/2*np.log(2*np.pi) - jnp.log(C).sum()
		return - d/2*np.log(2*np.pi) - jnp.log(C).sum()/2.

	print('mvn', mvn.logpdf(X, mean, jnp.diag(cov)))
	print('mine', diag_mvn_logpdf(X, mean, cov))
	print('incorrect', norm_const(cov))
	print('correct chol', norm_const_2(jnp.diag(cov)))
	# print(new_norm_const(jnp.diag(cov)))
	print('new',new_norm_const(cov))
