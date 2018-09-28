"""
Optimizer for molecular geometries in terms of "expected information gain"
"""

import os
import sys
import shutil
import numpy as np
import scipy.sparse.linalg
import scipy.optimize
from khan.training.trainer_multi_tower import TrainerMultiTower, initialize_module
from khan.data.dataset import RawDataset
from khan.model import activations
import data_utils
import tensorflow as tf
#tf.enable_eager_execution() # TODO: switch to graph once speed is needed

BOHR_PER_ANGSTROM = 0.52917721092
kT = 0.6 # kcal/mol


def load_NN_models(filenames, sessions):
    # files are expected to be in npz format
    # sess should be an initialized tensorflow session
    models = []
    for n, filename in enumerate(filenames):
        sess = sessions[n]
        towers = ["/cpu:0"]  # consider using ["/cpu:%d" % n] ?
        #layers = (128, 128, 64, 1)
        layers = tuple([256]*4 + [1])
        activation_fn = activations.get_fn_by_name('waterslide')
        with tf.variable_scope("model%d" % n): # each trainer needs its own scope
        #if True:
            trainer = TrainerMultiTower(
                sess,
                towers=towers,
                precision=tf.float64,
                layer_sizes=layers,
                activation_fn=activation_fn,
                fit_charges=False,
            )
            trainer.load_numpy(filename, strict=False)#, ignore="model%d/" % n)
        models.append(trainer)
    return models


def model_E_and_grad(xyz, elements, model):
    # xyz and elements must be merged into [element, x, y, z]
    # model must be a Trainer object
    nn_atom_types = [data_utils.atomic_number_to_atom_id(elem) for elem in elements]
    xyz = [[i]+list(xx) for i, xx in zip(nn_atom_types, xyz)]
    #print('in model_E_and_grad, xyz =', xyz)
    rd = RawDataset([xyz], [0.0])
    energy = float(model.predict(rd)[0])
    self_interaction = np.sum(data_utils.selfIxnNrgWB97X_631gdp[t] for t in nn_atom_types)
    energy += self_interaction
    gradient = list(model.coordinate_gradients(rd))[0]
    natoms, ndim = gradient.shape
    gradient = gradient.reshape(natoms*ndim)
    gradient *= BOHR_PER_ANGSTROM
    return energy, gradient


def opt_E_func(x_flat, elements, model):
    # basic energy function to optimize, for finding min_E
    xyz = np.reshape(x_flat, (-1, 3))
    #print('in opt_E_func, xyz =', xyz, 'elements =', elements)
    E, grad = model_E_and_grad(xyz, elements, model)
    return E#, grad


def opt_P_func(x_flat, elements, min_Es, models, calc_grad=False):
    # more advanced objective function, giving mean probability
    # for initial probability maximization, balancing between models
    xyz = np.reshape(x_flat, (-1, 3))
    Es = []
    dEdx, dEdy, dEdz = [], [], []
    for min_E, model in zip(min_Es, models):
        E, grad = model_E_and_grad(xyz, elements, model)
        rel_E = (E - min_E) / kT
        Es.append(rel_E)
        dEdx.append(grad / kT)
    Es = np.array(Es)
    exp_Es = np.exp(-Es)
    P = np.mean(exp_Es)
    if not calc_grad:
        return -P
    # grad not implemented yet
    
    
def opt_info_func(x_flat, elements, min_Es, models, calc_grad=False):
    # TODO: add "similarity to points already guessed" as a metric here
    # The use of a similarity metric implies we have a prior about which
    # points are likely to be the same as each other
    xyz = np.reshape(x_flat, (-1, 3))
    Es = []
    dEdx, dEdy, dEdz = [], [], []
    for min_E, model in zip(min_Es, models):
        E, grad = model_E_and_grad(xyz, elements, model)
        rel_E = (E - min_E) / kT
        Es.append(rel_E)
        dEdx.append(grad / kT)
    Es = np.array(Es)
    exp_Es = np.exp(-Es)
    P = np.mean(exp_Es)
    # The following assumes that model errors are roughly Gaussian
    # If Laplacian, 2*(mean absolute difference from median) would be used
    # to approximate the spread of energies, instead of stddev
    # which would be more robust but harder to find the gradient of
    if False:
        info = np.log(np.std(Es)) # Gaussian assumption
    else:
        info = np.log( np.sum(np.abs( Es - np.median(Es) ) ) ) # Laplacian assumption
    uniqueness = 1.0
    expected_info = P * info * uniqueness
    if not calc_grad:
        return -expected_info

    # note, something is wrong with this gradient:
    # does not agree with the numerical gradient
    dEdx = np.array(dEdx)
    dEdx = dEdx.T  # shape = (3*n_atoms) x (n_models)
    dP_dx = -np.mean(dEdx * exp_Es, axis=1) # shape = (3*n_atoms
    dinfo_dx = (np.mean(Es * dEdx) - np.mean(Es) * np.mean(dEdx, axis=1)) / info
    #return -P, -dP_dx
    #return -info, -dinfo_dx
    dexpected_info_dx = dP_dx*info + P*dinfo_dx # shape = (3*n_atoms

    return -expected_info, -dexpected_info_dx 
    

def run_opt(xyz, models):
    print(xyz)
    # xyz should be in the form [ [element x y z], ... ]
    elements = [row[0] for row in xyz]
    x0 = [row[1:] for row in xyz]
    x0 = np.reshape(x0, len(x0) * 3 )  # flatten coords: [x1 y1 z1 x2 y2 z2 ... ]
    print(elements)
    print(x0)
    # get min E for each model
    min_Es = []
    for model in models:
        # optimize energy for this model
        print('Trying initial energy optimization for model', model)
        result = scipy.optimize.fmin_l_bfgs_b(opt_E_func, x0, args=(elements, model), iprint=0, factr=1e3, approx_grad=True)
        min_x, min_E, success = result
        min_Es.append(min_E)
        print('model min_E =', min_E)

    # maximize mean probability at start (to provide a good start point)
    x0 = min_x
    result = scipy.optimize.fmin_l_bfgs_b(opt_P_func, x0, args=(elements, min_Es, models), iprint=0, factr=1e3, approx_grad=True)
    x0, P0, success = result
    print('P0 =', -P0)
    print('Max P geometry:', x0)
    # run scipy optimize
    print( 'Initial expected info =', -opt_info_func(x0, elements, min_Es, models, False))
    result = scipy.optimize.fmin_l_bfgs_b(opt_info_func, x0, args=(elements, min_Es, models), iprint=1, factr=1e1, approx_grad=True)
    x_final, fun_final, success = result
    print('Final expected info =', -fun_final)
    print('Final xyz coords:')
    xyz = np.reshape(x_final, (-1, 3))
    for el, xx in zip(elements, xyz):
        print(el, xx)
    
    # for testing gradients
    print_gradient_per_atom = False
    if print_gradient_per_atom:
        analytical_grads = np.reshape(opt_info_func(x0, True)[1], (-1, 3))
        numerical_grads = np.reshape(success['grad'], (-1, 3))
        print('Gradient per atom:')
        print('index    element    grad_analytic    grad_numerical')
        for i, atom, analytical_grad, numerical_grad in zip(range(len(st.atom)), st.atom, analytical_grads, numerical_grads):
            print(i+1, atom.element, analytical_grad, numerical_grad)


def test_nn_opt():
    # Load NN models from existing files
    # Todo: store layer sizes and activations in file too
    xyz_filename = sys.argv[1]
    with open(xyz_filename) as xyz_file:
        test_xyz = [ [float(s) for s in line.split()] for line in list(xyz_file)[2:] ]
    #test_xyz = [ [8, 0.0, 0.0, 0.0], [1, 1.0, 0.0, 0.0], [1, 0.0, 1.0, 0.0] ]
    test_xyz = np.array(test_xyz)
    model_filenames = ['sep28_%d/save/best.npz' % i for i in [1,2,3]]
    # load models
    lib_path = os.path.abspath('../gpu_featurizer/ani.so')
    initialize_module(lib_path)
    config = tf.ConfigProto(allow_soft_placement=True)
    #with tf.Session(config=config) as sess
    sessions = [ tf.Session(config=config) for name in model_filenames ]
    models = load_NN_models(model_filenames, sessions)
    run_opt(test_xyz, models)


test_nn_opt()
