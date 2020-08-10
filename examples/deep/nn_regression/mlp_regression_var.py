import numpy as np
np.random.seed(0)

import haiku as hk

import jax.numpy as jnp
from jax.experimental import optimizers
import jax

from tqdm import tqdm, trange
from matplotlib import pyplot as plt

from jax_bayes.variational import diagonal_mvn_fns
from jax_bayes.utils import confidence_bands

def build_dataset():
    n_train, n_test, d = 100, 100, 1
    xlims = [-1.0, 5.0]
    x_train = np.random.rand(n_train, d) * (xlims[1] - xlims[0]) + xlims[0]
    x_test = np.random.rand(n_test, d) * (xlims[1] - xlims[0]) + xlims[0]

    target_func = lambda t: (np.log(t + 100.0) * np.sin(1.0 * np.pi*t)) + 0.1 * t

    y_train = target_func(x_train)
    y_test = target_func(x_test)

    y_train += np.random.randn(*x_train.shape) * (1.0 * (x_train + 2.0)**0.5)

    return (x_train, y_train), (x_test, y_test)

def net_fn(x):
    sig = 4.0
    rbf = lambda x: jnp.exp(-x**2)#deep basis function model
    activation = rbf
    # activation = jnp.tanh
    mlp = hk.Sequential([
        hk.Linear(128, w_init=hk.initializers.Constant(0), 
                       b_init=hk.initializers.Constant(0)), 
        activation, 
        hk.Linear(1,   w_init=hk.initializers.Constant(0), 
                       b_init=hk.initializers.Constant(0))
        ])
    return mlp(x)

def main():
    # ======= Setup =======
    xy_train, xy_test = build_dataset()
    (x_train, y_train), (x_test, y_test) = xy_train, xy_test

    lr = 5e-2
    reg = 0.1 #this is it
    lik_var = 0.1

    net = hk.transform(net_fn)
    params = net.init(jax.random.PRNGKey(42), x_train)

    seed = 0
    key = jax.random.PRNGKey(seed)
    vf  = diagonal_mvn_fns(key, init_sigma = 0.1)
    var_params, var_keys = vf.init(params)
    
    opt_init, opt_update, opt_get_params = optimizers.adam(lr)
    opt_state = opt_init(var_params)

    @jax.jit
    def logprob(params, xy):
        """ log posterior, assuming 
        P(params) ~ N(0,eta)
        P(y|x, params) ~ N(f(x;params), lik_var)
        """
        x, y = xy
        preds = net.apply(params, None, x)
        log_prior = - reg * sum(jnp.sum(jnp.square(p)) 
                            for p in jax.tree_leaves(params))
        log_lik = - jnp.mean(jnp.square(preds - y)) / lik_var
        return log_lik + log_prior

    num_samples = 50

    @jax.jit
    def bbvi_step(i, opt_state, var_keys, batch):
        var_params = opt_get_params(opt_state)
        logp = lambda p: logprob(p, batch)
        logp = jax.vmap(logp)

        var_keys = vf.next_key(var_keys) #generate one key to use now
        next_keys = vf.next_key(var_keys) #generate one to return

        def elbo(p, keys):
            samples_state, _ = vf.sample(0, num_samples, keys, p)
            samples = vf.get_samples(samples_state)
            
            # 'stick the landing' ELBO estimator see https://arxiv.org/pdf/1703.09194.pdf
            return jnp.mean(logp(samples) - 
                    vf.evaluate(samples_state, jax.lax.stop_gradient(p)))

        obj = lambda p: - elbo(p, var_keys)

        loss, dlambda = jax.value_and_grad(obj)(var_params)
        opt_state = opt_update(i, dlambda, opt_state)
        return opt_state, loss, next_keys
    
    # ======== Optimization =========

    hist = []
    for step in trange(2000):
        if step % 250 == 0:
            var_params = opt_get_params(opt_state)
            samples_state, _ = vf.sample(0, num_samples, var_keys, var_params)
            param_samples = vf.get_samples(samples_state)
            
            logp = lambda params:logprob(params, xy_train)
            train_logp = jnp.mean(jax.vmap(logp)(param_samples))

            _elbo = jnp.mean(jax.vmap(logp)(param_samples) -
                             vf.evaluate(samples_state, var_params))
            
            logp = lambda params:logprob(params, xy_test)
            test_logp = jnp.mean(jax.vmap(logp)(param_samples))
            print(f"step = {step}"
                  f" | train logp = {train_logp:.3f}"
                  f" | test logp = {test_logp:.3f}"
                  f" | train elbo = {_elbo:.3f}")
        
        opt_state, loss, var_keys = bbvi_step(step, opt_state, var_keys, xy_train)
        hist.append(loss)

    # generate the final samples
    var_params = opt_get_params(opt_state)
    samples_state, _ = vf.sample(0, 10, var_keys, var_params)
    param_samples = vf.get_samples(samples_state)

    logp =  lambda params:logprob(params, xy_train)
    final_logp = jnp.mean(jax.vmap(logp)(param_samples))
    print(f'final logp = {final_logp:.3f}')

    # =========== Plotting ===========
    plot_inputs = np.linspace(-1, 10, num=600).reshape(-1,1)
    outputs = jax.vmap(lambda params: net.apply(params, None, plot_inputs))(param_samples)
    
    lower, upper = confidence_bands(outputs.squeeze(-1).T)
    
    f, axes = plt.subplots(2)
    
    ax = axes[0]
    ax.plot(hist)

    ax = axes[1]
    ax.plot(x_train.ravel(), y_train.ravel(), 'bx', color='green')
    ax.plot(x_test.ravel(), y_test.ravel(), 'bx', color='red')
    for i in range(outputs.shape[0]):
        ax.plot(plot_inputs, outputs[i], alpha=0.25)
    ax.plot(plot_inputs, np.mean(outputs[:, :, 0].T, axis=1), color='black', linewidth=1.0)
    ax.fill_between(plot_inputs.squeeze(-1), lower, upper, alpha=0.75)

    plt.show()

if __name__ == '__main__':
    main()