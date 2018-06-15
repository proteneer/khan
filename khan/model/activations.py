import tensorflow as tf
import functools
# collection of useful activation functions
# (ytz tip): use functools.partial if you need to fill-in the kwargs
def celu(A, alpha=0.1):
    posA = tf.cast(tf.greater_equal(A, 0), A.dtype) * A
    negA = tf.cast(tf.less(A, 0), A.dtype) * A
    alpha = alpha
    return posA + alpha * (tf.exp(negA/alpha) - 1)

def gaussian(A):
	return tf.exp(-1*tf.pow(A, 2))
