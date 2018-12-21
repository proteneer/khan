""" Functions related to using lad charges to compute long range coulomb interactions """
import itertools
import numpy as np
from khan.utils.helpers import atom_id_to_atomic_number
from khan.lad import lad
from khan.utils.constants import ANGSTROM_IN_BOHR
import os
from multiprocessing import cpu_count
import time

def softmax(x, xcut=4.0, a=10.0):
    """
    Softmax function which approximates max(x, xcut)
    a controls the rapidity of the switch
    """
    dx = x - xcut
    exponent = np.exp(-a*dx)
    return (x + xcut*exponent) / (1.0 + exponent)

def coulomb(X, q):
    """
    Return the long range coulomb interaction 
    for a geometry X (in khan format)
    and a corresponding list of charges
    """ 
    ener = 0.0
    natoms = len(X)
    for i, j in itertools.combinations(range(natoms), 2):
        dX = X[i][1:] - X[j][1:]
        d = np.sqrt(np.dot(dX, dX))
        ener += q[i]* q[j] / softmax(d)

    return ener * ANGSTROM_IN_BOHR

def lad_charges(X, lad_params, reference_lads):
    """
    Get the lad charges for a molecule

    X: the molecule
    lad_params: parameters for LAD construction
    reference_lads: reference LADs
    """
    atom_ids = [atom[0] for atom in X]
    atomic_numbers = list(map(atom_id_to_atomic_number, atom_ids))
    carts = [atom[1:] for atom in X]
    charges = lad.interpolate_charges(
        atomic_numbers,
        carts,
        lad_params,
        reference_lads,
        neutralize=True
    )
    return charges

def remove_coulomb(molecules, energies, lad_params, reference_lads, pool=None):
    """
    Remove coulomb energy from dataset
    """

    start_time = time.time()
    print("Removing LR coulomb energy from %d datapoints..." % len(energies))
    ndata = len(energies)

    func = map
    if pool is not None:
        print("Using thread pool")
        func = pool.map
    else:
        print("executing in serial")

    ce = CoulombEnergy(lad_params, reference_lads)
    coulomb_energies = func(ce.compute, molecules)

    for idx, correction in enumerate(coulomb_energies):
        energies[idx] -= correction

    elapsed_time = time.time() - start_time

    print("Time for Coulomb removal: %.2f s" % elapsed_time)

class CoulombEnergy(object):
    """
    Computes the coulomb energy of a molecule 
    """
    def __init__(self, lad_params, reference_lads):
        self._lad_params = lad_params
        self._reference_lads = reference_lads
        
    def compute(self, molecule):
        """
        Compute the coulomb energy for a single molecule
        """ 
        q = lad_charges(molecule, self._lad_params, self._reference_lads)
        ener = coulomb(molecule, q)
        return ener
        
