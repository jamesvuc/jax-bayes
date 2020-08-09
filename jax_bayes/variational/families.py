import math

import jax
import jax.numpy as jnp

import numpy as np

from .variational_family import variational_family

def elbo_reparam(logprob, samples, var_approx, var_params):
    return jnp.mean(logprob(samples) - var_approx(samples, var_params))

def gaussian_elbo_reparam(logprob, samples, var_params):
    return jnp.mean(logprob(samples)) + diag_mvn_entropy(var_params)

def elbo_noscore(logprob, samples, var_approx, var_params):
    var_params = jax.lax.stop_gradient(var_params)
    return - jnp.mean(logprob(samples) - var_approx(samples, var_params))

def diag_mvn_logpdf(x, mean, diag_cov):
    """ Returns the log_pdf of x under a MVN with diagonal covariance without
    storing the full covariance for O(N) storage instead of O(N^2).
    """
    n = mean.shape[-1]
    y = x - mean
    tmp = jnp.einsum('...i,i->...i', y, 1./diag_cov)
    return (-1/2 * jnp.einsum('...i,...i->...', y, tmp)
            - n/2 * np.log(2*np.pi) - jnp.log(diag_cov).sum()/2.)

def diag_mvn_entropy(logcov):
    d = logcov.shape[0]
    return 0.5 * d * (1.0 + np.log(2*np.pi)) + jnp.sum(logcov)

@variational_family
def diagonal_mvn_fns(base_key, mean_stddev=1.0, init_sigma=1.0, eps=1e-6):
    """ Constructs functions for a VariationalFamily object using a
    diagonal multivariate normal (equivalent to mean-field) variational family.

    Args:
        base_key: jax.random.PRNGKey used to seed the randomness for the algorithm
        mean_stddev: standard deviation of mean parameter initialization
        init_sigma: starting standard deviation of the mean field parameters
        eps: tolerance for the log-covariance

    Returns:
        init, sample, evaluate, get_samples, next_key, entropy, base_key
        functions for the VariationalFamily (to be tree-ified in the decorator).
        Entropy can be a dummy function, the others are needed for VI.
    """
    def init(x0, key):
        next_key, key = jax.random.split(key)
        mean = jax.random.normal(key, x0.shape) * mean_stddev
        logcov = jnp.zeros_like(x0) + math.log(init_sigma)

        return (mean, logcov), next_key

    #TODO: Remove arg 'i'
    def sample(i, num_samples, key, params):
        """ sample from q( |params) """
        key, next_key = jax.random.split(key)
        mean, logcov = params

        shape = (num_samples,) +  mean.shape
        Z = jax.random.normal(key, shape)

        return Z * jnp.exp(logcov) + mean, next_key

    def evaluate(inputs, params):
        """ evaluate logq( |params) """
        mean, logcov = params
        mean, logcov = mean.reshape(-1), logcov.reshape(-1)
        inputs = inputs.reshape(inputs.shape[0], -1)

        cov = jnp.exp(logcov) + eps
        return diag_mvn_logpdf(inputs, mean, cov)

    def get_samples(samples):
        return samples

    def next_key(key):
        _, new_key = jax.random.split(key)
        return new_key

    def entropy(params):
        _, logcov = params
        return diag_mvn_entropy(logcov)

    return init, sample, evaluate, get_samples, next_key, entropy, base_key
