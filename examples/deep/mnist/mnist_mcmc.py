import haiku as hk

import jax.numpy as jnp
from jax.experimental import optimizers
import jax

import sys, os, math, time
import numpy as onp
from tqdm import tqdm, trange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow_datasets as tfds

sys.path.append('../../..')
from jax_bayes.mcmc import rk_langevin_fns, mala_fns, hmc_fns, rms_langevin_fns

# sampler_fns = langevin_fns
# sampler_fns = rk_langevin_fns
sampler_fns = rms_langevin_fns
# sampler_fns = hmc_fns

""" to test:
- more samples
- original init
- other algs
"""

def load_dataset(split, is_training, batch_size):
	ds = tfds.load('mnist:3.*.*', split=split).cache().repeat()
	if is_training:
		ds = ds.shuffle(10 * batch_size, seed=0)
	ds = ds.batch(batch_size)
	return tfds.as_numpy(ds)


def net_fn(batch):
	""" Standard LeNet-300-100 MLP 
		
		In the jax way, we define a function that operates on the inputs (data)
	"""
	x = batch["image"].astype(jnp.float32) / 255.
	# sig = 0.7
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
	# lr = 1e-3
	# lr = 1e-2
	lr = 5e-3
	# reg = 1e-3
	# reg = 1e-5
	reg = 1e-4
	# reg = 0.0

	#hk.transform returns a Transformed object with methods init and apply
	net = hk.transform(net_fn)

	def loss(params, batch):
		logits = net.apply(params, batch)
		labels = jax.nn.one_hot(batch['label'], 10)

		l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) 
							for p in jax.tree_leaves(params))
		softmax_crossent = - jnp.mean(labels * jax.nn.log_softmax(logits))

		return softmax_crossent + reg * l2_loss

	logprob = lambda p,b : - loss(p, b)

	@jax.jit
	def accuracy(params, batch):
		# preds = net.apply(params, batch)
		# return jnp.mean(jnp.argmax(preds, axis=-1) == batch['label'])
		pred_fn = lambda p:net.apply(p, batch) 
		pred_fn = jax.vmap(pred_fn)
		preds = jnp.mean(pred_fn(params), axis=0)
		return jnp.mean(jnp.argmax(preds, axis=-1) == batch['label'])

	@jax.jit
	def mcmc_step(i, sampler_state, sampler_keys, batch):
		params = sampler_get_params(sampler_state)
		logp = lambda p:logprob(p,  batch) #can make this 1-line?
		fx, dx = jax.vmap(jax.value_and_grad(logp))(params)

		sampler_prop_state, new_keys = sampler_propose(i, dx, sampler_state, 
														sampler_keys)

		fx_prop, dx_prop = fx, dx
		if sampler_fns in [mala_fns, rk_langevin_fns]:
			prop_params = sampler_get_params(sampler_prop_state)
			fx_prop, dx_prop = jax.vmap(jax.value_and_grad(logp))(prop_params)
		
		elif sampler_fns == hmc_fns:
			# for j in range(5): #5 iterations of the leapfrog integrator
			for j in range(2):	
				sampler_prop_state, new_keys = \
					sampler_propose(i, dx_prop, sampler_prop_state, new_keys)
				
				prop_params = sampler_get_params(sampler_prop_state)
				fx_prop, dx_prop = jax.vmap(jax.value_and_grad(logp))(prop_params)

		sampler_state, new_keys = sampler_update(i, fx, fx_prop, 
												 dx, sampler_state, 
												 dx_prop, sampler_prop_state, 
												 new_keys)
		
		return jnp.mean(fx), sampler_state, new_keys

	# train_batches = load_dataset("train", is_training=True, batch_size=1_000)
	train_batches = load_dataset("train", is_training=True, batch_size=1_000)
	val_batches = load_dataset("train", is_training=False, batch_size=10_000)
	test_batches = load_dataset("test", is_training=False, batch_size=10_000)

	#need to draw batch (using next(train)) to get shapes
	# I would prefer torch-style define-then-run rather than define-by-run
	params = net.init(jax.random.PRNGKey(42), next(train_batches))
	# seed = int(time.time() * 1000)
	seed = 0
	key = jax.random.PRNGKey(seed)
	sampler_init, sampler_propose, sampler_update, sampler_get_params = \
		sampler_fns(key, num_samples=10, step_size=lr, init_stddev=0.1)
		# sampler_fns(key, num_samples=2, step_size=lr, init_stddev=0.5)
	
	# --------------
	# initialize samples w/ net initializer using different keys
	# num_samples = 10
	# sampler_init, sampler_propose, sampler_update, sampler_get_params = \
	# 	sampler_fns(key, num_samples=-1, step_size=lr, init_stddev=0.1)

	# param_keys = jax.random.split(jax.random.PRNGKey(42), num_samples)
	# init_batch = next(train_batches)
	# net_initializer = lambda k: net.init(k, init_batch)
	# params = jax.vmap(net_initializer)(param_keys)
	# --------------

	sampler_state, sampler_keys = sampler_init(params)

	# for step in trange(5_001):
	for step in trange(10_001):
		if step % 500 == 1:
			params = sampler_get_params(sampler_state)
			val_acc = accuracy(params, next(val_batches))
			test_acc = accuracy(params, next(test_batches))
			print(f"step = {step}"
				  f" | train logprob ={train_logprob:.3f}"
				  f" | val acc = {val_acc:.3f}"
				  f" | test acc = {test_acc:.3f}")
		
		train_logprob, sampler_state, sampler_keys = \
			mcmc_step(step, sampler_state, sampler_keys, next(train_batches))

	print('done')



if __name__ == '__main__':
	main()


