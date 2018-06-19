import tensorflow as tf
import functools
import sys

# collection of useful activation functions
# (ytz tip): use functools.partial if you need to fill-in the kwargs

def get_fn_by_name(fn_name, *args):
	"""
	Get an activation function by name using default parameters.

	Example usage:

	afn = activations.get_fn_by_name("celu") # default alpha=0.1
	afn = activations.get_fn_by_name("celu", 0.2) # override
    afn = activations.get_fn_by_name("normal", 0.5, 0.2) # extra mean, std

	Parameters
	----------
	fn_name: str
		Name of the function to retrieve. eg. "celu" will return this
		module's celu function.

	args: extra fn arguments
		Optional arguments (non-keyword) to be passed into the function.

	..note:: This is mainly used when you need to dynamically switch
		activation functions for things like hyperparameter smashing via
		the command line. This is not recommend for production use.

	"""
	this_mod = sys.modules[__name__]
	fn = getattr(this_mod, fn_name)
	return functools.partial(fn, *args)

def celu(A, alpha=0.1):
    posA = tf.cast(tf.greater_equal(A, 0), A.dtype) * A
    negA = tf.cast(tf.less(A, 0), A.dtype) * A
    return posA + alpha * (tf.exp(negA/alpha) - 1)

def normal(A, mean=0.0, std=1.0):
	return tf.exp(-tf.pow((A - mean)/std, 2)/2.0)

def gaussian(A):
	return tf.exp(-1*tf.pow(A, 2))
