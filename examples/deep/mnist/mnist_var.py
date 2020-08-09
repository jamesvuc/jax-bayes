import haiku as hk

import jax.numpy as jnp
from jax.experimental import optimizers
import jax

import sys, os, math, time
import numpy as onp
from tqdm import tqdm, trange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow_datasets as tfds

sys.path.append('../..')
from jax_bayes.variational import diagonal_mvn_fns

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
	vf = diagonal_mvn_fns(key, init_sigma = 0.1)
	var_params, var_keys = vf.init(params)

	lr = 1e-3
	# lr = 5e-4
	opt_init, opt_update, opt_get_params = optimizers.adam(lr)
	opt_state = opt_init(var_params)

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
		pred_probs = jnp.mean(pred_fn(params), axis=0)#average out the model params
		return jnp.mean(jnp.argmax(pred_probs, axis=-1) == batch['label'])


	num_samples = 10#25
	@jax.jit
	def bbvi_step(i, opt_state, var_keys, batch):
		var_params = opt_get_params(opt_state)
		logp = lambda p: logprob(p, batch)
		# fx, dx = jax.vmap(jax.value_and_grad(logp))(var_params)
		logp = jax.vmap(logp)

		var_keys = vf.next_key(var_keys) #generate one key to use now
		next_keys = vf.next_key(var_keys) #generate one to return

		def elbo(p, keys):
			samples_state, _ = vf.sample(0, num_samples, keys, p)
			samples = vf.get_samples(samples_state)
			return - jnp.mean(logp(samples) - vf.evaluate(samples_state, p))
			# return - jnp.mean(logp(samples) 
			# 		- var_eval(samples_state, jax.lax.stop_gradient(p)))

		obj = lambda p: elbo(p, var_keys)

		_loss, dlambda = jax.value_and_grad(obj)(var_params)
		opt_state = opt_update(i, dlambda, opt_state)
		return opt_state, _loss, next_keys

	@jax.jit
	def eval_step(opt_state, var_keys, batch):
		var_params = opt_get_params(opt_state)
		samples_state, _ = vf.sample(0, num_samples, var_keys, var_params)
		params = vf.get_samples(samples_state)
		
		val_acc = accuracy(params, next(val_batches))
		test_acc = accuracy(params, next(test_batches))

		return dict(val_acc=val_acc, test_acc=test_acc)

	# for step in trange(5_001):
	for step in trange(10_001):
		if step % 100 == 1:
			# var_params = opt_get_params(opt_state)
			# samples_state, _ = var_sample(0, num_samples, var_keys, var_params)
			# params = var_get_samples(samples_state)
			
			# val_acc = accuracy(params, next(val_batches))
			# test_acc = accuracy(params, next(test_batches))
			metrics = eval_step(opt_state, var_keys, var_params)
			print(f"step = {step}"
				  f" | train loss ={_loss:.3f}"
				  f" | val acc = {metrics['val_acc']:.3f}"
				  f" | test acc = {metrics['test_acc']:.3f}")
				  # f" | val acc = {val_acc:.3f}"
				  # f" | test acc = {test_acc:.3f}")
		
		opt_state, _loss, var_keys = \
			bbvi_step(step, opt_state, var_keys, next(train_batches))

	print('done')



if __name__ == '__main__':
	main()


