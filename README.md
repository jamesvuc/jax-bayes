# jax-bayes
Bayesian inference with Python and Jax.

## Overview
jax-bayes is designed to accelerate research in high-dimensional Bayesian inference, specifically for deep neural networks. This library is built on [Jax](https://github.com/google/jax) which has a number of features that make it ideal for working with high-dimensional probability distributions including: functional programming, automatic differentiation, and automatic vectorization, accelerator-agnostic backend via [XLA](https://www.tensorflow.org/xla), etc... (more discussion below).

jax-bayes supports two different methods for sampling from high-dimensional inference: **Markov Chain Monte Carlo** (MCMC) and **Variational Inference** (VI). The former iterates a Markov chain which has an invariant measure distribution (approximately) equal to the target distribution, whereas the latter finds the closest (in some sense) distribution in a parameterized family to the target distribution. 

jax-bayes allows you to define a sampler (MCMC or VI) that operates on arrays, and uses some very effective meta-programming in the form of decorators to "tree-ify" these methods to operate on arbitrary containers. This is the approach taken in ``jax.experimental.optimizers``, and we have essentially adapted it to support MCMC and VI. This flexible approach allows you to *use neural networks from other JAX-based libraries for bayesian ML* (our examples use Deemind's [Haiku](https://github.com/deepmind/dm-haiku) library for building neural networks).

*TODO: Add quickstart with the langevin algorithm*

## Installation
jax-bayes requires jax>=0.1.74 and jaxlib>=0.1.15 as separate dependencies, since jaxlib needs to be installed with accelerator (CPU)

Assuming you have jax + jaxlib installed, install via pip:
```
pip install git+https://github.com/jamesvuc/jax-bayes
```

## Package Description
- ``jax_bayes.mcmc`` contains the MCMC functionality. It provides:
    - ``jax_bayes.mcmc.sampler`` which is the decorator that "tree-ifies" a sampler's methods. A sampler is defined as a callable returning a tuple of functions
    ```python
        def sampler(*args, **kwargs):
            ...
            return init, propose, update, get_params
    ```
    where the returned functions have specific signatures. 
    - ``jax_bayes.mcmc.langevin_fns`` is a sampler definition which implements the Unadjusted Langevin Algorithm
    - ``jax_bayes.mcmc.mala_fns`` is a sampler definition which implements the Metropolis Adjusted Langevin Algorithm
    - ``jax_bayes.mcmc.rk_langevin_fns`` is a sampler definition which implements a stochastic Runge Kutta solver for the continuous-time Langevin dyanmics
    -  ``jax_bayes.mcmc.hmc_fns`` is a sampler definition which implements the Hamitonian Monte Carlo algorithm
    - ``jax_bayes.mcmc.rms_langevin_fns`` is a sampler which implements a preconditioned Langevin algorithm using the exponential moving root-mean-square estimate of the gradient as the preconditionner matrix (like RMSProp)
    - ``jax_bayes.mcmc.rwmh_fns`` is a sampler which implements the Random Walk Metropolis Hastings Algorithm.
    - ``jax_bayes.mcmc.bb_mcmc`` is a function which puts the sampler functions into a "black-box" function suitable for sampling from simple densities (e.g. without sampling batches).
- ``jax_bayes.variational`` contains the variational inference functionality. It provides:
    - ``jax_bayes.variational.variational_family`` which is a decorator that tree-ifies the variational family's methods. A variational family is defined as a callable returning a tuple of functions
    ```python
        def variational_family(*args, **kwargs):
            ...
            return init, sample, evaluate, get_samples, next_key, entropy
    ```
    where the returned functions have specific signatures. The returned object is not, however, a tree-ified collection of functions but a class that contains these functions
    - ``jax_bayes.variational.diag_mvn_fns`` is a variational family definition which implements the diagonal multivariate gaussian family

## Examples

## Design Philosophy