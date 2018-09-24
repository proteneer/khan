"""
Functions for fitting force field parameters given energy and geometry data

Copyright Schrodinger LLC, All Rights Reserved.
"""

import os
import shutil
import copy
from multiprocessing import Pool, cpu_count
import numpy as np
import scipy.sparse  # for storing sparse matrixes
import scipy.sparse.linalg  # the main solver
import scipy.optimize  # for post-solve solution polishing
import sklearn.preprocessing  # for matrix preprocessing
from schrodinger import structure
from schrodinger.infra import mm
from schrodinger.utils import fileutils
from schrodinger.application.ffbuilder import common

COUL_CUTOFF = 9999.0  # distance cutoff in Angstrom, an arbitrary large value
CONV_ENER = 1e-12  # energy convergence, kcal/mol (1e-12 is nigh-unreachable)
CONV_GRAD = 1e-4  # force convergence, kcal/mol/Angstrom - the real threshold
MAX_STEP = 1500  # how many steps to take in minimization if conv not reached.


def setup_opls(st, mmffld_handle):
    mm.mmffld_setOption(mmffld_handle, mm.MMFfldOption_SYS_CUTOFF, 0,
                        COUL_CUTOFF, "")  # set cutoff to a large value
    mm.mmffld_setOption(mmffld_handle, mm.MMFfldOption_MIN_CONV_ENER, 0,
                        CONV_ENER, "")  # set energy convergence threshold
    mm.mmffld_setOption(mmffld_handle, mm.MMFfldOption_MIN_CONV_GRAD, 0,
                        CONV_GRAD, "")  # set gradient convergence threshold
    mm.mmffld_setOption(mmffld_handle, mm.MMFfldOption_MIN_LS_METHOD,
                        mm.MMFfldMinEnergyLS, 0.0, "")  # line-search by energy
    mm.mmffld_setOption(mmffld_handle, mm.MMFfldOption_MIN_MAX_STEP, 1500,
                        0.0, "")  # set max_step to 1500
    # enterMol performs the typing (a lengthy operation, by MM standards)
    mm.mmffld_enterMol(mmffld_handle, st)


def opls_energy_and_grad(mmffld_handle):
    '''
    Returns the energy and gradient for the molecule loaded in mmffld_handle
    Returned grad is a flat array: [x1, y1, z1, x2, y2, z2, ... ]
    '''
    forces, energy_components = mm.mmffld_getEnergyForce(mmffld_handle)
    energy = energy_components[0]
    grad = np.array([-f for f in forces])
    return energy, grad


def model_E_and_grad(xyz, model):
    mmffld_handle, st, model_params = model
    st.setXYZ(xyz)
    #dielectric = model_params
    #mm.mmffld_setOption(mmffld_handle, mm.MMFfldOption_ENE_DIELECTRIC, 0, dielectric*5, "")
    E, grad = opls_energy_and_grad(mmffld_handle)

    A, B = model_params
    r = st.measure(11,27)
    E += A/r**12 - B/r**6

    return E, grad


def test_opt():
    st = next(structure.StructureReader('test.mae'))
    with common.opls_force_field(no_cm1a_bcc=1) as mmffld_handle: # note, no fitted charges
        setup_opls(st, mmffld_handle)
        mm.mmffld_minimize(mmffld_handle)
        # energy function to optimize, for finding min_E
        def opt_E_func(x_flat, model_params):
            xyz = np.reshape(x_flat, (-1, 3))
            E, grad = model_E_and_grad(xyz, (mmffld_handle, st, model_params))
            return E#, grad
        
        # flatten coords for scipy
        x0 = np.reshape( st.getXYZ(), len(st.getXYZ())*3 )
        models = [ (i*1e5, i*1e4) for i in range(2)] # dummy models
        min_Es = []
        # get min per model
        for model_params in models:
            # optimize energy for this model
            result = scipy.optimize.fmin_l_bfgs_b(opt_E_func, x0, args=(model_params,), iprint=0, factr=1e1, approx_grad=True)
            min_x, min_E, success = result
            min_Es.append(min_E)
            print('model', model_params, '=> min_E', min_E)

        # for initial probability maximization, balancing between models
        def opt_P_func(x_flat, calc_grad=False):
            xyz = np.reshape(x_flat, (-1, 3))
            Es = []
            kT = 10.0 # kcal/mol
            dEdx, dEdy, dEdz = [], [], []
            for min_E, model_params in zip(min_Es, models):
                E, grad = model_E_and_grad(xyz, (mmffld_handle, st, model_params))
                rel_E = (E - min_E)/kT
                Es.append(rel_E)
                dEdx.append(grad/kT)
            Es = np.array(Es)
            exp_Es = np.exp(-Es)
            P = np.mean(exp_Es)
            if not calc_grad:
                return -P

        def opt_info_func(x_flat, calc_grad=False):
            # TODO: add "similarity to points already guessed" as a metric here
            # The use of a similarity metric implies we have a prior about which
            # points are likely to be the same as each other
            xyz = np.reshape(x_flat, (-1, 3))
            Es = []
            kT = 10.0 # kcal/mol
            dEdx, dEdy, dEdz = [], [], []
            for min_E, model_params in zip(min_Es, models):
                E, grad = model_E_and_grad(xyz, (mmffld_handle, st, model_params))
                rel_E = (E - min_E)/kT
                Es.append(rel_E)
                dEdx.append(grad/kT)
            Es = np.array(Es)
            exp_Es = np.exp(-Es)
            P = np.mean(exp_Es)
            info = np.std(Es)
            expected_info = P*info
            if not calc_grad:
                #return -P
                #return -info
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

        # maximize starting probability (to provide a good start point)
        x0 = min_x
        result = scipy.optimize.fmin_l_bfgs_b(opt_P_func, x0, iprint=0, factr=1e1, approx_grad=True)
        x0, P0, success = result
        print('P0 =', -P0)
        # run scipy optimize
        print( 'Initial expected info =', -opt_info_func(x0, False))
        result = scipy.optimize.fmin_l_bfgs_b(opt_info_func, x0, iprint=0, factr=1e1, approx_grad=True)
        x_final, fun_final, success = result
        print( 'Final expected info =', -fun_final)
        # save new geom
        st.write('opt_info.mae')

        print_gradient_per_atom = False
        if print_gradient_per_atom:
            analytical_grads = np.reshape(opt_info_func(x0, True)[1], (-1, 3))
            numerical_grads = np.reshape(success['grad'], (-1, 3))
            print('Gradient per atom:')
            print('index    element    grad_analytic    grad_numerical')
            for i, atom, analytical_grad, numerical_grad in zip(range(len(st.atom)), st.atom, analytical_grads, numerical_grads):
                print(i+1, atom.element, analytical_grad, numerical_grad)


test_opt()

