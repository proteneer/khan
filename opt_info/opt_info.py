"""
Functions for fitting force field parameters given energy and geometry data

Copyright Schrodinger LLC, All Rights Reserved.
"""

import os
import shutil
import numpy as np
import scipy.sparse.linalg
import scipy.optimize
from khan.training.trainer_multi_tower import TrainerMultiTower, initialize_module
from khan.data.dataset import RawDataset
from khan.model import activations
import data_utils
import tensorflow as tf

BOHR_PER_ANGSTROM = 0.52917721092
kT = 10.0 # kcal/mol


def load_NN_models(filenames=['example.npz']):
    models = []
    for n, filename in enumerate(filenames):
        towers = ["/cpu:0"]  # consider using ["/cpu:%d" % n] ?
        layers = (128, 128, 64, 1)
        trainer = TrainerMultiTower(
            sess,
            towers=towers,
            precision=tf.float32,
            layer_sizes=layers,
            activation_fn=activations.waterslide,
            fit_charges=False,
        )
        trainer.load_numpy(filename, strict=False)
        models.append(trainer)
    return models


def model_E_and_grad(xyz, model):
    # xyz must be element, x, y, z
    # model must be a Trainer object
    rd = RawDataset([xyz], [0.0])
    energy = float(model.predict(rd)[0])
    self_interaction = sum(data_utils.selfIxnNrgWB97X_631gdp[atom[0]] for atom in xyz)
    energy += self_interaction
    gradient = list(model.coordinate_gradients(rd))[0]
    natoms, ndim = gradient.shape
    gradient = gradient.reshape(natoms*ndim)
    gradient *= BOHR_PER_ANGSTROM
    return energy, gradient


def opt_E_func(x_flat, model_params):
    # basic energy function to optimize, for finding min_E
    xyz = np.reshape(x_flat, (-1, 3))
    E, grad = model_E_and_grad(xyz, model)
    return E#, grad


def opt_P_func(x_flat, min_Es, models, calc_grad=False):
    # more advanced objective function, giving mean probability
    # for initial probability maximization, balancing between models
    xyz = np.reshape(x_flat, (-1, 4))
    Es = []
    dEdx, dEdy, dEdz = [], [], []
    for min_E, model_params in zip(min_Es, models):
        E, grad = model_E_and_grad(xyz, model)
        rel_E = (E - min_E) / kT
        Es.append(rel_E)
        dEdx.append(grad / kT)
    Es = np.array(Es)
    exp_Es = np.exp(-Es)
    P = np.mean(exp_Es)
    if not calc_grad:
        return -P
    # grad not implemented yet
    
    
def opt_info_func(x_flat, min_Es, models, calc_grad=False):
    # TODO: add "similarity to points already guessed" as a metric here
    # The use of a similarity metric implies we have a prior about which
    # points are likely to be the same as each other
    xyz = np.reshape(x_flat, (-1, 3))
    Es = []
    dEdx, dEdy, dEdz = [], [], []
    for min_E, model_params in zip(min_Es, models):
        E, grad = model_E_and_grad(xyz, (mmffld_handle, st, model_params))
        rel_E = (E - min_E) / kT
        Es.append(rel_E)
        dEdx.append(grad / kT)
    Es = np.array(Es)
    exp_Es = np.exp(-Es)
    P = np.mean(exp_Es)
    info = np.std(Es)
    expected_info = P*info
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
    

def run_opt():
    xyz = 'Load me from a file'
    models = load_NN_models(filenames=['example1.npz', 'example2.npz'])
    x0 = np.reshape(xyz, len(xyz) * 4 )  # flatten coords: [i1 x1 y1 z1 i2 x2 y2 z2 ... ]
    # get min E for each model
    min_Es = []
    for model in models:
        # optimize energy for this model
        result = scipy.optimize.fmin_l_bfgs_b(opt_E_func, x0, args=(model,), iprint=0, factr=1e1, approx_grad=True)
        min_x, min_E, success = result
        min_Es.append(min_E)
        print('model', model_params, '=> min_E', min_E)

    # maximize mean probability at start (to provide a good start point)
    x0 = min_x
    result = scipy.optimize.fmin_l_bfgs_b(opt_P_func, x0, args=(min_Es, models), iprint=0, factr=1e1, approx_grad=True)
    x0, P0, success = result
    print('P0 =', -P0)
    # run scipy optimize
    print( 'Initial expected info =', -opt_info_func(x0, False))
    result = scipy.optimize.fmin_l_bfgs_b(opt_info_func, x0, args=(min_Es, models), iprint=0, factr=1e1, approx_grad=True)
    x_final, fun_final, success = result
    print( 'Final expected info =', -fun_final)
    xyz = np.reshape(x_final, (-1, 4))
    print('xyz =', xyz)
    
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
    lib_path = os.path.abspath('khan/gpu_featurizer/ani_cpu.so')
    initialize_module(lib_path)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        run_opt()

test_nn_opt()
