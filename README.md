# jax-bayes
High-dimensional Bayesian inference with Python and Jax.

## Overview
jax-bayes is designed to accelerate research in high-dimensional Bayesian inference, specifically for deep neural networks. It is built on [Jax](https://github.com/google/jax) 

jax-bayes supports two different methods for sampling from high-dimensional inference: 
- **Markov Chain Monte Carlo** (MCMC) which iterates a Markov chain which has an invariant measure distribution (approximately) equal to the target distribution
- **Variational Inference** (VI): which finds the closest (in some sense) distribution in a parameterized family to the target distribution. 

jax-bayes allows you to  **"bring your own JAX-based network to the Bayesian ML party"** by providing samplers that operate on arbitrary data structures of JAX arrays and JAX transformations. You can also define your own sampler in terms of JAX arrays and lift them to general-purpose samplers (using the same approach as in [``jax.experimental.optimizers``](https://jax.readthedocs.io/en/latest/_modules/jax/experimental/optimizers.html))

### Quickstart
You can easily modify this [Haiku quickstart example](https://github.com/deepmind/dm-haiku#quickstart) to support bayesian inference:
```python
# ---- From the Haiku Quickstart ----
import jax.numpy as jnp
import haiku as hk

def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

def logprob_fn(batch):

    mlp = hk.Sequential([
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(10),
    ])
    logits = mlp(batch['images'])
    return jnp.mean(softmax_cross_entropy(logits, batch['labels']))

logprob = hk.transform(logprob_fn)

# ---- With jax-bayes ---- 

#instantiate the sampler
key = jax.random.PRNGKey(0)
from jax_bayes.mcmc import langevin_fns
init, propose, update, get_params = langevin_fns(key, lr=1e-3)

#define the mcmc step
@jax.jit
def mcmc_step(state, keys, batch):
    params = get_params(state)
    batch_logprob = lambda p: logprob.apply(p, None, batch)
    
    #use vmap + grad to compute per-sample gradients
    g = jax.vmap(jax.grad(batch_logprob))(params)

    #omiting some unused arguments for this example
    propose_state, new_keys = sampler_propose(g, state, keys, ...)
    next_state, new_keys = sampler_update(propose_state, new_keys, ...)

    return next_state, new_keys

#initialize the sampler state
params = logprob.init(jax.random.PRNGKey(1), next(dataset))
sampler_state, keys = init(params)

#run the mcmc algorithm
for i in range(1000):
    sampler_state, keys = mcmc_step(sampler_state, keys, next(dataset))

# extract your samples
sampled_params = get_params(sampler_state)
```

### Logits != Uncertainty
Sometimes we want our neural networks to say "I don't know" (think self-driving cars, machine translation, etc) but, as well-illustrated out in http://proceedings.mlr.press/v48/gal16.pdf, the logits of a neural network should not serve a substitute for uncertainty. This library allows you to model *weight uncertainty* about the data by sampling from the posterior rather than optimizing it. You can also take advantge of occam's razor and other benefits of Bayesian statistics.

## Installation
jax-bayes requires jax>=0.1.74 and jaxlib>=0.1.15 as separate dependencies, since jaxlib needs to be [installed](https://github.com/google/jax#pip-installation) for the accelerator (CPU / GPU / TPU).

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
    - A bunch of samplers:
        - ``jax_bayes.mcmc.langevin_fns`` (Unadjusted Langevin Algorithm)
        - ``jax_bayes.mcmc.mala_fns`` (Metropolis Adjusted Langevin Algorithm)
        - ``jax_bayes.mcmc.rk_langevin_fns`` (stochastic Runge Kutta solver for the continuous-time Langevin dyanmics)
        -  ``jax_bayes.mcmc.hmc_fns`` (Hamitonian Monte Carlo algorithm)
        - ``jax_bayes.mcmc.rms_langevin_fns`` (preconditioned Langevin algorithm using the smoothed root-mean-square estimate of the gradient as the preconditionner matrix (like RMSProp))
        - ``jax_bayes.mcmc.rwmh_fns`` implements (Random Walk Metropolis Hastings Algorithm.)
    - ``jax_bayes.mcmc.bb_mcmc`` wraps a given sampler into a "black-box" function suitable for sampling from simple densities (e.g. without sampling batches).
- ``jax_bayes.variational`` contains the variational inference functionality. It provides:
    - ``jax_bayes.variational.variational_family`` which is a decorator that tree-ifies the variational family's methods. A variational family is defined as a callable returning a tuple of functions
    ```python
        def variational_family(*args, **kwargs):
            ...
            return init, sample, evaluate, get_samples, next_key, entropy
    ```
    where the returned functions have specific signatures. The returned object is not, however, a tree-ified collection of functions but a class that contains these functions
    - ``jax_bayes.variational.diag_mvn_fns`` (diagonal multivariate gaussian family)

## Examples
We have provided some diverse examples, some of which are under active development --- see ``examples/`` for more details. As an overview:
1. Shallow examples for sampling from regular probability distributions using MCMC and VI (such as):
![alt text](https://github.com/jamesvuc/jax-bayes/blob/master/assets/mcmc_2d.png "Logo Title Text 1")
2. Deep examples for doing deep Bayesian ML (mostly with Colab)
    1. Neural Network Regession
    2. MNIST with 300-100-10 MLP
    3. CIFAR10 with a CNN
    4. Attention-based Neural Machine Translation
![alt text](https://github.com/jamesvuc/jax-bayes/blob/master/assets/nn_regression_mcmc.png "Logo Title Text 1")

<!-- ## Design Philosophy -->

<!-- which has a number of features that make it ideal for working with high-dimensional probability distributions including: functional programming, automatic differentiation, and automatic vectorization, accelerator-agnostic backend via [XLA](https://www.tensorflow.org/xla), etc... (more discussion below).

jax-bayes allows you to define a sampler (MCMC or VI) that operates on arrays, and uses some very effective meta-programming in the form of decorators to "tree-ify" these methods to operate on arbitrary containers. This is the approach taken in ``jax.experimental.optimizers``, and we have essentially adapted it to support MCMC and VI. This flexible approach allows you to *use neural networks from other JAX-based libraries for bayesian ML* (our examples use Deemind's [Haiku](https://github.com/deepmind/dm-haiku) library for building neural networks).
 -->