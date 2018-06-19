import tensorflow as tf
import functools

DEFAULT_ACTIVATION = "CELU"

# collection of useful activation functions
# (ytz tip): use functools.partial if you need to fill-in the kwargs
def celu(A, alpha=0.1):
    posA = tf.cast(tf.greater_equal(A, 0), A.dtype) * A
    negA = tf.cast(tf.less(A, 0), A.dtype) * A
    return posA + alpha * (tf.exp(negA/alpha) - 1)

def gaussian(A):
	return tf.exp(-1*tf.pow(A, 2))

def softplus(A, shift=0.69314718056):
    return tf.nn.softplus(A) - shift

ACTIVATION_FUNCTIONS = {
    "CELU": celu,
    "SELU": tf.nn.selu,
    "LEAKY_RELU": functools.partial(tf.nn.leaky_relu, alpha=0.2),
    "GAUSSIAN": gaussian,
    "SOFTPLUS": softplus
}

