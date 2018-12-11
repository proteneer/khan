import tensorflow as tf
import functools
import sys
import inspect

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

def get_all_fn_names():
    """
    Return a list of names of all defined activation functions in
    this file.

    Returns
    -------
    list of str
        List of names of activation functions.

    ..note:: This is mainly used when you need to dynamically switch
        activation functions for things like hyperparameter smashing via
        the command line. This is not recommend for production use.

    """
    all_names = []
    for fn_name, fn_handle in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        if fn_name[:4] == "get_":
            continue
        all_names.append(fn_name)
    return all_names

# Note to maintainers: do not prefix your activation functions with the
# get_ string. This prefix is used by the get_all_activation_fn_names
# to strip out utility functions.

def celu(A, alpha=0.5):
    # tensorflow's elu function thankfully doesn't implement the alpha multiplier before the elu
    # so it's trivial to implement celu using an elu without having to implement a slow switch function
    return alpha*tf.nn.elu(A/alpha)

def normal(A, mean=0.0, std=1.0):
    return tf.exp(-tf.pow((A - mean)/std, 2)/2.0)

def gaussian(A):
    return tf.exp(-1*tf.pow(A, 2))

# softplus - ln(2), such that the function passes through the origin (helps initialization)
def softplus_origin(A):
    return tf.nn.softplus(A) - 0.69314718056

# waterslide, identity at the origin, gradient is periodic: sin(x)+1
def waterslide(A):
    return A + 1 - tf.cos(A)
