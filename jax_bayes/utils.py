import jax
import jax.numpy as jnp

def certainty_acc(pp, targets, cert_threshold=0.5):
    """ Calculates the accuracy-at-certainty from the predictive probabilites pp
    on the targets.

    Args:
        pp: (batch_size, n_classes) array of probabilities
        targets: (batch_size, n_calsses) array of label class indices
        cert_threhsold: (float) minimum probability for making a prediction

    Returns:
        accuracy at certainty, indicies of those prediction instances for which
        the model is certain.
    """
    preds = jnp.argmax(pp, axis=1)
    pred_probs = jnp.max(pp, axis=1)

    certain_idxs = pred_probs >= cert_threshold
    acc_at_certainty = jnp.mean(targets[certain_idxs] == preds[certain_idxs])

    return acc_at_certainty, certain_idxs

@jax.jit
@jax.vmap
def entropy(p):
    """ computes discrete Shannon entropy.
    p: (n_classes,) array of probabilities corresponding to each class
    """
    p += 1e-12 #tolerance to avoid nans while ensuring 0log(0) = 0
    return - jnp.sum(p * jnp.log(p))

def confidence_bands(y, sample_axis=-1):
    """ Computes confidence bands for samples y.

    Args:
        y: array of samples.

    Returns:
        mean and standard deviation along the axes not equal to sample_axis.
    """
    m = jnp.mean(y, axis=sample_axis)
    s = jnp.std(y, axis=sample_axis)
    return m - s, m + s
