import haiku as hk

import jax.numpy as jnp
# from jax import grad, vmap, jit
from jax.experimental import optimizers
# from jax.experimental import optix 
import jax
import numpy as onp
from tqdm import tqdm, trange

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow_datasets as tfds

def load_dataset(split, is_training, batch_size):
	ds = tfds.load('mnist:3.*.*', split=split).cache().repeat()
	if is_training:
		ds = ds.shuffle(10 * batch_size, seed=0)
	ds = ds.batch(batch_size)
	return tfds.as_numpy(ds)

""" - the main difference between jax and other DSLs is that jax is functional.
	- Therefore a model is a function, not an object like nn.Module. 
	- Haiku will allow you to build init(...) and apply(...) functions out of 
		model specifications easily.

	- it seems that a common technique is local inheritance though...
"""

def net_fn(batch):
	""" Standard LeNet-300-100 MLP 
		
		In the jax way, we define a function that operates on the inputs (data)
	"""
	x = batch["image"].astype(jnp.float32) / 255.
	mlp = hk.Sequential([
		hk.Flatten(),
		hk.Linear(300), jax.nn.relu, 
		hk.Linear(100), jax.nn.relu, 
		hk.Linear(10)])

	#where are the parameters? hk.Params object?
	return mlp(x)

def main():
	# lr = 1e-3
	lr = 5e-3
	reg = 1e-4

	#hk.transform returns a Transformed object with methods init and apply
	net = hk.transform(net_fn)
	# adam_init, adam_update, adam_get_params = optimizers.adam(lr)
	adam_init, adam_update, adam_get_params = optimizers.sgd(lr)
	# opt = optix.adam(lr)

	def loss(params, batch):
		logits = net.apply(params, batch)
		labels = jax.nn.one_hot(batch['label'], 10)

		l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) 
							for p in jax.tree_leaves(params))
		# softmax_crossent = - jnp.sum(labels * jax.nn.log_softmax(logits))
		softmax_crossent = - jnp.mean(labels * jax.nn.log_softmax(logits))

		return softmax_crossent + reg * l2_loss

	@jax.jit
	def accuracy(params, batch):
		preds = net.apply(params, batch)
		return jnp.mean(jnp.argmax(preds, axis=-1) == batch['label'])

	@jax.jit
	def train_step(i, opt_state, batch):
		params = adam_get_params(opt_state)
		dx = jax.grad(loss)(params, batch)
		opt_state = adam_update(i, dx, opt_state)
		return opt_state

	# def ema_step(a)

	train_batches = load_dataset("train", is_training=True, batch_size=1_000)
	val_batches = load_dataset("train", is_training=False, batch_size=10_000)
	test_batches = load_dataset("test", is_training=False, batch_size=10_000)

	#need to draw batch (using next(train)) to get shapes
	# I would prefer torch-style define-then-run rather than define-by-run
	params = net.init(jax.random.PRNGKey(42), next(train_batches))
	
	opt_state = adam_init(params)
	# for step in trange(5_001):
	for step in trange(10_001):
		if step % 1_000 == 0:
			params = adam_get_params(opt_state)
			val_acc = accuracy(params, next(val_batches))
			test_acc = accuracy(params, next(test_batches))
			print(f"step = {step}"
				  f" | val acc = {val_acc:.3f}"
				  f" | test acc = {test_acc:.3f}")
		
		opt_state = train_step(step, opt_state, next(train_batches))

	print('done')



if __name__ == '__main__':
	main()


