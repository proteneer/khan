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

BOHR_PER_ANGSTROM = 0.52917721092
kT = 0.001 # in Hartree


def load_NN_models(filenames, sessions):
    # files are expected to be in npz format
    # sess should be an initialized tensorflow session
    models = []
    for n, filename in enumerate(filenames):
        sess = sessions[n]
        towers = ["/cpu:0"]  # consider using ["/cpu:%d" % n] ?
        layers = tuple([256]*4 + [1])
        activation_fn = activations.get_fn_by_name('waterslide')
        with tf.variable_scope("model%d" % n): # each trainer needs its own scope
            trainer = TrainerMultiTower(
                sess,
                towers=towers,
                precision=tf.float64,
                layer_sizes=layers,
                activation_fn=activation_fn,
                fit_charges=False,
            )
            trainer.load_numpy(filename, strict=False)
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
    gradient = gradient.reshape(gradient.size)
    gradient *= BOHR_PER_ANGSTROM
    return energy/kT, gradient/kT # return E in units of kT, gradient in units of kT/Angstrom


def opt_E_func(x_flat, elements, model):
    # basic energy function to optimize, for finding min_E
    xyz = np.reshape(x_flat, (-1, 3))
    #print('in opt_E_func, xyz =', xyz, 'elements =', elements)
    E, grad = model_E_and_grad(xyz, elements, model)
    return E, grad


def opt_P_func(x_flat, elements, min_Es, models, calc_grad=False):
    # more advanced objective function, giving mean probability
    # for initial probability maximization, balancing between models
    xyz = np.reshape(x_flat, (-1, 3))
    Es, dEdx = [], []
    for min_E, model in zip(min_Es, models):
        E, grad = model_E_and_grad(xyz, elements, model)
        rel_E = (E - min_E)
        Es.append(rel_E)
        dEdx.append(grad)
    Es = np.array(Es)
    exp_Es = np.exp(-Es)
    P = np.mean(exp_Es)
    if not calc_grad:
        return -P
    # grad not implemented yet
    
    
def coul_mat(xyz):
    # xyz is n_atoms x 3
    mat = np.zeros(len(xyz), len(xyz))
    for i, xi in enumerate(xyz):
        for j, xj in enumerate(xyz):
            if i == j: continue
            mat[i][j] = 1 / np.linalg.norm(xi-xj)
    return mat
    
def opt_info_func(x_flat, elements, min_Es, models, calc_grad=False):
    # TODO: add "similarity to points already guessed" as a metric here
    # The use of a similarity metric implies we have a prior about which
    # points are likely to be the same as each other
    n_results = len(x_flat) // len(elements)
    xyzs = np.reshape(x_flat, (-1, 3))
    expected_info_gain_per_point = []
    for n in range(n_results):
        Es, dEdx = [], []
        xyz = xyzs[n*len(elements) : (n+1)*len(elements)]
        for min_E, model in zip(min_Es, models):
            E, grad = model_E_and_grad(xyz, elements, model)
            rel_E = (E - min_E)
            Es.append(rel_E)
            dEdx.append(grad)
        Es = np.array(Es)
        exp_Es = np.exp(-Es)
        P = np.mean(exp_Es)
        if False:
            info = np.log(np.std(Es)) # Gaussian assumption
        else:
            info = np.log( np.sum(np.abs( Es - np.median(Es) ) ) ) # Laplacian assumption
        expected_info_gain_per_point.append( P * info )
    for n1 in range(n_results):
        similarity_sum = 0.0
        for n2 in range(n_results):
           if n1==n2: continue
           xyz1 = xyzs[n1*len(elements) : (n1+1)*len(elements)]
           xyz2 = xyzs[n2*len(elements) : (n2+1)*len(elements)]
           coul_mat1 = coul_mat(xyz1)
           coul_mat2 = coul_mat(xyz2)
           rms_diff = np.sqrt( np.sum((dist_mat1 - dist_mat2)**2) / dist_mat1.size )
           length_scale = 1.0 # wild guess
           similarity_sum += np.exp( -rms_diff * length_scale )
        # note, max of similarity_sum is n_results
        uniqueness = 1.0 - similarity_sum / n_results
        expected_info_gain_per_point[n1] *= uniqueness
    if not calc_grad:
        return -sum(expected_info_gain_per_point) # - because we want to maximize, not minimize
    

def run_opt(xyz, models, n_results=1):
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
        result = scipy.optimize.fmin_l_bfgs_b(opt_E_func, x0, args=(elements, model), iprint=0, factr=1e3, pgtol=1e-5*kT)#, approx_grad=True, epsilon=1e-5)
        min_x, min_E, success = result
        min_Es.append(min_E)
        print('model min_E =', min_E)

    # maximize mean probability at start (to provide a good start point)
    x0 = min_x
    result = scipy.optimize.fmin_l_bfgs_b(opt_P_func, x0, args=(elements, min_Es, models), iprint=0, factr=1e3, approx_grad=True, epsilon=1e-5, pgtol=1e-5*kT)
    x0, P0, success = result
    print('P0 =', -P0)
    print('Max P geometry:', x0)
    # run scipy optimize
    print( 'Initial expected info =', -opt_info_func(x0, elements, min_Es, models, False))
    xs = np.concatenate( [x0] * n_results ) # split starting geom into n_results starting geoms
    xs += np.random.normal(scale=0.01, size=xs.shape) # randomize starting positions a little
    result = scipy.optimize.fmin_l_bfgs_b(opt_info_func, xs, args=(elements, min_Es, models), iprint=1, factr=1e1, approx_grad=True, epsilon=1e-5, pgtol=1e-5*kT)
    x_final, fun_final, success = result
    print('Final expected info =', -fun_final)
    print('Final xyz coords:')
    xyz = np.reshape(x_final, (-1, 3))
    element_names = {1.0:'H', 6.0:'C', 7.0:'N', 8.0:'O'}
    for n in n_results:
        print(len(elements))
        print('Result', n)
        for el, xx in zip(elements, xyz[n*len(elements) : (n+1)*len(elements)]):
            print(element_names[el], '%f %f %f' % tuple(xx))
    
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
    global kT
    xyz_filename = sys.argv[1]
    with open(xyz_filename) as xyz_file:
        test_xyz = [ [float(s) for s in line.split()] for line in list(xyz_file)[2:] ]
    kT = float(sys.argv[2])
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
