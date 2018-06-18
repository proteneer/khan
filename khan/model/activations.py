import tensorflow as tf
import functools
# collection of useful activation functions
# (ytz tip): use functools.partial if you need to fill-in the kwargs
def celu(A, alpha=0.1):
    posA = tf.cast(tf.greater_equal(A, 0), A.dtype) * A
    negA = tf.cast(tf.less(A, 0), A.dtype) * A
    return posA + alpha * (tf.exp(negA/alpha) - 1)

# Roitberg's original activation function
def gaussian(A):
    return 2*tf.exp(-0.5*tf.pow(A, 2))

# softplus - ln(2), such that the function passes through the origin (helps initialization)
def softplus_origin(A):
    return tf.nn.softplus(A) - 0.69314718056

