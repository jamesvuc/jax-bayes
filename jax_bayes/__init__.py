""" jax-bayes is a bayesian inference library for JAX """

from jax_bayes import mcmc
from jax_bayes import variational
from jax_bayes import utils

__version__ = "0.1.0"

__all__ = (
    "mcmc",
    "variational",
    "utils"
)