
# Examples

We provide 2 levels of examples:
1. `examples/shallow`: These demonstrate the different samplers on 1- and 2- dimenisonal probability distributions (i.e. not neural networks). 

2. `examples/deep`: These showcase how jax-bayes can be used for deep Bayesian ML. The goal is to allow one to compare different inference techniques apply to some standard problems in ML:
	1. neural network regression
	2. MNIST
	3. CIFAR10
	4. Neural Machine Translation

Some of these are nontrivial to implement with current Bayesian methods.

*current status*:

 example | optimization | MCMC | VI
:--:|:--:|:--:|:--:
nn regression | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
MNIST | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
CIFAR10 | :heavy_check_mark: | :heavy_check_mark:(broken)
NMT | :heavy_check_mark:

<!-- TODO: Talk about bayesian stuff, integrating out parameters rather than optimizing them. These provide integration algorithms. Use mathjax -->

## Visualizations
Here are visualizations from some of the examples:

### `examples/shallow/mcmc_1d.py`
![](https://github.com/jamesvuc/jax-bayes/blob/master/assets/mcmc_1d.png "1d MCMC")

### `examples/shallow/mcmc_2d.py`

![](https://github.com/jamesvuc/jax-bayes/blob/master/assets/mcmc_2d.png "2d MCMC")

### `examples/shallow/variational_2d.py`
![](https://github.com/jamesvuc/jax-bayes/blob/master/assets/variational_2d.png "2d variational")

### `examples/deep/nn_regression/mlp_regression_mcmc.py`
![](https://github.com/jamesvuc/jax-bayes/blob/master/assets/nn_regression_mcmc.png "2d variational")