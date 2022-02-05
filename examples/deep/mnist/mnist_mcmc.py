import haiku as hk

import jax.numpy as jnp
from jax.experimental import optimizers
import jax

import jax_bayes
from jax_bayes.mcmc import mala_fns

import sys, os, math, time
import numpy as np
from tqdm import trange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow_datasets as tfds

from matplotlib import pyplot as plt

#load the data and create the model
def load_dataset(split, is_training, batch_size):
    ds = tfds.load('mnist:3.*.*', split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))

def net_fn(batch):
    """ Standard LeNet-300-100 MLP """
    x = batch["image"].astype(jnp.float32) / 255.

    # we initialize the model with zeros since we're going to construct intiial 
    # samples for the weights with additive Gaussian noise
    sig = 0.0
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(300, w_init=hk.initializers.RandomNormal(stddev=sig),
                       b_init=hk.initializers.RandomNormal(stddev=sig)), 
        jax.nn.relu, 
        hk.Linear(100, w_init=hk.initializers.RandomNormal(stddev=sig),
                       b_init=hk.initializers.RandomNormal(stddev=sig)), 
        jax.nn.relu, 
        hk.Linear(10,  w_init=hk.initializers.RandomNormal(stddev=sig),
                       b_init=hk.initializers.RandomNormal(stddev=sig))
        ])

    return mlp(x)


def main():
    #hyperparameters
    lr = 5e-3
    # lr = 1e-3
    reg = 1e-4
    num_samples = 100 # number of samples to approximate the posterior
    init_stddev = 5.0 # initial distribution for the samples will be N(0, 0.1)
    train_batch_size = 1_000
    eval_batch_size = 10_000

    #instantiate the model --- same as regular case
    net = hk.transform(net_fn)

    #build the sampler instead of optimizer
    sampler_fns = mala_fns
    seed = 0
    key = jax.random.PRNGKey(seed)
    sampler_init, sampler_propose, sampler_accept, sampler_update, sampler_get_params = \
        sampler_fns(key, num_samples=num_samples, step_size=lr, init_stddev=init_stddev,
        noise_scale=0.1)


    # loss is the same as the regular case! This is because in regular ML, we're minimizing
    # the negative log-posterior logP(params | data) = logP(data | params) + logP(params) + constant
    # i.e. finding the MAP estimate.
    def loss(params, batch):
        logits = net.apply(params, jax.random.PRNGKey(0), batch)
        labels = jax.nn.one_hot(batch['label'], 10)
        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) 
                    for p in jax.tree_leaves(params))
        softmax_crossent = - jnp.mean(labels * jax.nn.log_softmax(logits))

        return softmax_crossent + reg * l2_loss

    #the log-probability is the negative of the loss
    logprob = lambda p,b : - loss(p, b)

    @jax.jit
    def accuracy(params, batch):
        #auto-vectorize over the samples of params! only in JAX...
        pred_fn = jax.vmap(net.apply, in_axes=(0, None, None))

        # this is crucial --- integrate (i.e. average) out the parameters
        all_logits = pred_fn(params, None, batch)
        probs = jnp.mean(jax.nn.softmax(all_logits, axis=-1), axis=0)

        return jnp.mean(jnp.argmax(probs, axis=-1) == batch['label'])

    #build the mcmc step. This is like the opimization step, but for sampling
    @jax.jit
    def mcmc_step(i, state, keys, batch):
        #extract parameters
        params = sampler_get_params(state)
        # rvals = sampler_get_params(state, idx=1)
        
        #form a partial eval of logprob on the data
        logp = lambda p: logprob(p,  batch) #can make this 1-line?
        
        # evaluate *per-sample* gradients
        fx, dx = jax.vmap(jax.value_and_grad(logp))(params)

        # generat proposal states for the Markov chains
        prop_state, new_keys = sampler_propose(i, dx, state, keys)
        
        #we don't need to re-compute gradients for the accept stage
        prop_params = sampler_get_params(prop_state)
        fx_prop, dx_prop = jax.vmap(jax.value_and_grad(logp))(prop_params)

        # generate the acceptance indices from the Metropolis-Hastings
        # accept-reject step
        accept_idxs, keys = sampler_accept(
            i, fx, fx_prop, dx, state, dx_prop, prop_state, keys
        )

        # update the sampler state based on the acceptance acceptance indices
        state, keys = sampler_update(
            i, accept_idxs, dx, state, dx_prop, prop_state, keys
        )
        
        return fx, state, new_keys

    # load the data into memory and create batch iterators
    train_batches = load_dataset("train", is_training=True, batch_size=train_batch_size)
    val_batches = load_dataset("train", is_training=False, batch_size=eval_batch_size)
    test_batches = load_dataset("test", is_training=False, batch_size=eval_batch_size)


    #get a single sample of the params using the normal hk.init(...)
    params = net.init(jax.random.PRNGKey(42), next(train_batches))

    # get a SamplerState object with `num_samples` params along dimension 0
    # generated by adding Gaussian noise (see sampler_fns(..., init_dist='normal'))
    sampler_state, sampler_keys = sampler_init(params)

    # iterate the the Markov chain
    for step in trange(2_501):
        train_logprobs, sampler_state, sampler_keys = \
            mcmc_step(step, sampler_state, sampler_keys, next(train_batches))
    
        if step % 500 == 0:
            params = sampler_get_params(sampler_state)
            val_acc = accuracy(params, next(val_batches))
            test_acc = accuracy(params, next(test_batches))
            print(f"step = {step}"
                f" | val acc = {val_acc:.3f}"
                f" | test acc = {test_acc:.3f}")
    
    def posterior_predictive(params, batch):
        pred_fn = lambda p:net.apply(p, None, batch) 
        pred_fn = jax.vmap(pred_fn)

        logit_samples = pred_fn(params) # n_samples x batch_size x n_classes
        pred_samples = jnp.argmax(logit_samples, axis=-1) #n_samples x batch_size

        n_classes = logit_samples.shape[-1]
        batch_size = logit_samples.shape[1]
        probs = np.zeros((batch_size, n_classes))
        for c in range(n_classes):
            idxs = pred_samples == c
            probs[:,c] = idxs.sum(axis=0)

        return probs / probs.sum(axis=1, keepdims=True)


    def do_analysis():
        test_data = next(test_batches)
        pred_fn = jax.vmap(net.apply, in_axes=(0, None, None))

        all_test_logits = pred_fn(params, None, test_data)
        probs = jnp.mean(jax.nn.softmax(all_test_logits, axis=-1), axis=0)
        correct_preds_mask = jnp.argmax(probs, axis=-1) == test_data['label']

        # pp = posterior_predictive(params, test_data)
        pp = probs
        entropies = jax_bayes.utils.entropy(pp)

        correct_ent = entropies[correct_preds_mask]
        incorrect_ent = entropies[~correct_preds_mask]

        mean_correct_ent = jnp.mean(correct_ent)
        mean_incorrect_ent = jnp.mean(incorrect_ent)

        plt.hist(correct_ent, alpha=0.3, label='correct', density=True)
        plt.hist(incorrect_ent, alpha=0.3, label='incorrect', density=True)
        plt.axvline(x=mean_correct_ent, color='blue', label='mean correct')
        plt.axvline(x=mean_incorrect_ent, color='orange', label='mean incorrect')
        plt.legend()
        plt.xlabel("entropy")
        plt.ylabel("histogram density")
        plt.title("posterior predictive entropy of correct vs incorrect predictions")
        plt.show()

    do_analysis()
    plt.show()


if __name__ == '__main__':
    main()