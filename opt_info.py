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
NN_LAYERS = tuple([256]*4 + [1])
NN_ACTIVATION = 'waterslide'

def load_NN_models(filenames, sess):
    # files are expected to be in npz format
    # sess should be an initialized tensorflow session
    models = []
    for n, filename in enumerate(filenames):
        towers = ["/cpu:0"]  # consider using ["/cpu:%d" % n] ?
        layers = NN_LAYERS
        activation_fn = activations.get_fn_by_name(NN_ACTIVATION)
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
    #gradient = list(model.coordinate_gradients(rd))[0]
    #gradient = gradient.reshape(gradient.size)
    #gradient *= BOHR_PER_ANGSTROM
    return energy/kT, None  #gradient/kT # return E in units of kT, gradient in units of kT/Angstrom


def opt_E_func(x_flat, elements, model):
    # basic energy function to optimize, for finding min_E
    xyz = np.reshape(x_flat, (-1, 3))
    #print('in opt_E_func, xyz =', xyz, 'elements =', elements)
    E, grad = model_E_and_grad(xyz, elements, model)
    return E#, grad


def opt_P_func(x_flat, elements, min_Es, models, calc_grad=False):
    # A more advanced objective function, giving mean probability
    # for initial probability maximization, balancing between models
    # Other options: get probability as exp(-median(Es), mean(Es), or min(Es))
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
    else:
        grad_negative_P = np.dot(exp_Es, dEdx)
        return -P, grad_negative_P  # negative because we want to maximize, not minimize
    
    
def coul_mat(xyz):
    # xyz is n_atoms x 3
    mat = np.zeros((len(xyz), len(xyz)))
    for i, xi in enumerate(xyz):
        for j, xj in enumerate(xyz):
            if i == j: continue
            mat[i][j] = 1 / np.linalg.norm(xi-xj)
    return mat
   
    
def opt_info_func(x_flat, elements, min_Es, models, calc_grad=False):
    n_results = len(x_flat) // (len(elements)*3)  # how many distinct xyz systems we have
    xyzs = np.reshape(x_flat, (n_results, len(elements), 3))
    expected_info_gain_per_point = []
    grad_norm_per_point = []
    median_E_per_point = []
    for n, xyz in enumerate(xyzs):
        Es, dEdx = [], []
        for min_E, model in zip(min_Es, models):
            E, grad = model_E_and_grad(xyz, elements, model)
            rel_E = (E - min_E)
            Es.append(rel_E)
            dEdx.append(grad)
        Es = np.array(Es)
        exp_Es = np.exp(-Es)
        P = np.mean(exp_Es)
        inherent_variance = (0.001 / kT)**2  # in units of (Hartree/kT)^2
        ref_variance = (0.0005 / kT)**2  # in units of (kT)^2 - refers to the errors of the QM reference method
        if True:  # assume model Es are sampled from a Gaussian distribution with a certain mean and variance
            models_variance = np.var(Es, ddof=1) + inherent_variance
            # KL divergence between Gaussians with same mean, different variances:
            info = np.log( np.sqrt(models_variance/ref_variance) ) + (ref_variance/models_variance - 1)/2
        else: # Laplacian assumption? Not sure how variances add for Laplacians
            info = np.log( np.sqrt(inherent_variance + np.sum(np.abs( Es - np.median(Es) ) )**2) )
        expected_info_gain_per_point.append( P * info )
        #print('P, info = %.2g %.2g' % (P, info))
        #grad_norm_per_point.append(np.mean(np.linalg.norm(dEdx, axis=1)))
        median_E_per_point.append(np.median(Es))
    for n1, xyz1 in enumerate(xyzs):
        similarity_sum = 0.0
        for n2, xyz2 in enumerate(xyzs):
           if n1==n2: continue
           if False:
               coul_mat1, coul_mat2 = coul_mat(xyz1), coul_mat(xyz2)
               rms_diff = np.sqrt( np.sum((coul_mat1 - coul_mat2)**2) / coul_mat1.size )
               scale = 0.1# * grad_norm_per_point[n1] # bigger grad => need denser info
               #print('scale, rms_diff =', scale, rms_diff)
               similarity_sum += np.exp( -rms_diff * scale ) / n_results
           if True:
               similarity_sum += np.exp( -np.abs( median_E_per_point[n1] - median_E_per_point[n2] ) ) / n_results
        # note, max value of similarity_sum == 1
        uniqueness = 1.0 - similarity_sum
        #print('P, info, Es =', P, info, Es, uniqueness)
        expected_info_gain_per_point[n1] *= uniqueness
    if not calc_grad:
        return -np.sum(expected_info_gain_per_point) # negative because we want to maximize, not minimize


def run_opt(xyz, models, n_results=1):
    # xyz should be in the form [ [element x y z], ... ]
    elements = [row[0] for row in xyz]
    x0 = [row[1:] for row in xyz]
    x0 = np.reshape(x0, len(x0) * 3 )  # flatten coords: [x1 y1 z1 x2 y2 z2 ... ]
    # get min E for each model
    min_Es = []
    for model in models:
        # optimize energy for this model
        print('Trying initial energy optimization for model', model)
        result = scipy.optimize.fmin_l_bfgs_b(opt_E_func, x0, args=(elements, model), iprint=0, factr=1e3, pgtol=1e-6*kT, approx_grad=True, epsilon=1e-6)
        min_x, min_E, success = result
        #result = scipy.optimize.basinhopping(opt_E_func, x0, minimizer_kwargs={'args':(elements, model), 'method':'Nelder-Mead'}, niter=10, T=1.0, stepsize=1.0, disp=True) # T is in kT units, stepsize in Angstroms
        #min_x, min_E = result.x, result.fun
        min_Es.append(min_E)
        print('model min_E =', min_E)

    # maximize mean probability at start (to provide a good start point)
    x0 = min_x
    result = scipy.optimize.fmin_l_bfgs_b(opt_P_func, x0, args=(elements, min_Es, models), maxiter=50, iprint=0, factr=1e3, pgtol=1e-4*kT, approx_grad=True, epsilon=1e-6)
    x0, P0, success = result
    print('P0 =', -P0)
    print('Max P geometry:', '\n'.join([str(s) for s in zip(elements, np.reshape(x0, (-1, 3)))]))
    # run scipy optimize
    print( 'Initial expected info =', -opt_info_func(x0, elements, min_Es, models, False))
    #for n_results in range(1,20):
    if True:
        xs = np.concatenate( [x0] * n_results ) # split starting geom into n_results starting geoms
        xs += np.random.normal(scale=0.01, size=xs.shape) # randomize starting positions a little
        import time
        start = time.time()
        result = scipy.optimize.fmin_l_bfgs_b(opt_info_func, xs, args=(elements, min_Es, models), maxiter=300, iprint=0, factr=1e1, approx_grad=True, epsilon=1e-6, pgtol=1e-6*kT)
        x_final, fun_final, success = result
        #result = scipy.optimize.minimize(opt_info_func, xs, args=(elements, min_Es, models), method='Nelder-Mead', options={'maxiter':500})
        print('Info opt time:', time.time() - start, 'seconds', -fun_final)
    #x_final, fun_final = result.x, result.fun
    #result = scipy.optimize.basinhopping(opt_info_func, xs, minimizer_kwargs={'args':(elements, min_Es, models), 'method':'Nelder-Mead'}, niter=5, T=1.0, stepsize=1.0, disp=True) # T is in kT units, stepsize in Angstroms
    #x_final, fun_final = result.x, result.fun
    print('Final expected info =', -fun_final)
    xyzs = np.reshape(x0, (1, len(elements), 3))
    element_names = {1.0:'H', 6.0:'C', 7.0:'N', 8.0:'O'}
    print(len(elements))
    print('Max P geometry')
    for el, xx in zip(elements, xyzs[0]):
        print(element_names[el], '%f %f %f' % tuple(xx))
    xyzs = np.reshape(x_final, (n_results, len(elements), 3))
    element_names = {1.0:'H', 6.0:'C', 7.0:'N', 8.0:'O'}
    for n in range(n_results):
        print(len(elements))
        print('Max info', n)
        for el, xx in zip(elements, xyzs[n]):
            print(element_names[el], '%f %f %f' % tuple(xx))

            
def test_nn_opt(xyz_filename, kT, n_results):
    # xyz_filename = starting geometry
    # n_results = how many new geometries to deliver
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
    sess = tf.Session(config=config)
    models = load_NN_models(model_filenames, sess)
    run_opt(test_xyz, models, n_results)

#import profile
#profile.run('test_nn_opt()')
if __name__ == '__main__':
   xyz_filename = sys.argv[1]
   kT = float(sys.argv[2])
   n_results = int(sys.argv[3]) if len(sys.argv)>3 else 1
   test_nn_opt(xyz_filename, kT, n_results)

