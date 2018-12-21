import json
import os
import glob
import numpy as np
import sys

from itertools import zip_longest
from collections import OrderedDict, defaultdict, namedtuple

LADData = namedtuple("LADData", ["atomic_number", "charge", "mcn"])

# stolen from data_utils (not importable)
def atomic_number_to_atom_id(atno):
    """
    Return an atom index (ANI atom type)  given an atomic number
    atomic number must be convertable to an int
    """
    return {1: 0, 6: 1, 7: 2, 8: 3}[int(atno)]

LAD_PARAMS = {
    # Parameters for distance metric
    'wm': 4.0,  # MCN exponent
    'wq': 0.0,  # was 5.0 (not using here)

    # Extra parameters for overlap metric
    'sigmoid_s': 25.0,
    'sigmoid_c': 0.7,

    # Parameters for MCN definition
    'nested_MCN': False,
    'k1': 16.0,
    'k2': 5.0 / 3.0,

}

# Threshold for treating sum of LAD weights as zero.
ZERO_NORM = 1e-15

def interpolate_atomic_parameter(lad_instance, reference_data):
    """
    LAD Interpolation of a single atomic property

    :type lad_instance: LAD
    :param lad_instance: LAD for an atom
    :type reference_data: ReferenceLadData
    :param reference_data: holds reference lad data
    :type reference_charges: list of floats
    :param reference_charges: y values to interpolate
    :return the interpolated value
    """

    value = 0.0
    norm = 0.0

    return reference_data.interpolate(lad_instance)
        
def read_reference_lads(reference_data_file, params):
    """
    Read precomputed lad data

    This data is stored in a json file and is pre-generated
    :type reference_data_file: string
    :param reference_data_file: name of json file
    :type params: dict
    :param params: list of parameters for LAD construction
    :returnp a ReferenceLadData instance
    """
    assert reference_data_file.endswith(".json")
    
    with open(reference_data_file, "r") as fin:
        data = json.load(fin)

    lads = {}
    for example in data:
        lad_data = LADData(**example)
        lad_instance = LAD(
            params,
            lad_data.atomic_number,
            lad_data.charge,
            lad_data.mcn)
        lads.setdefault(lad_data.atomic_number, []).append((lad_data.charge, lad_instance))

    return ReferenceLadData(lads)

def interpolate_charges(
    atomic_numbers, carts, params, reference_data, neutralize=True):
    """
    Interpolate charges for a structure

    :type atomic_numbers: list of ints
    :param atomic_numbers: atomic numbers for each atom
    :type carts: numpy array
    :param carts: cartesian coordinates
    :type params: dict
    :param params: lad parameters
    :type reference_data: ReferenceLadData
    :param reference_data: reference lad data
    :type neutralize: boolean
    :param neutralize: neutralize the system
    :return a list of interpolated charges
    """
    natoms = len(atomic_numbers) 
    st_mcn = MCN(params, atomic_numbers, carts)
    charges = []
    for idx in range(natoms): 
        at_mcn = st_mcn[idx]
        at_lad = LAD(
            params,
            atomic_numbers[idx],
            0.0, 
            at_mcn)

        q = interpolate_atomic_parameter(
            at_lad, reference_data)

        charges.append(q) 

    if neutralize:
        total_q = sum(charges)
        correction = -total_q / natoms
        neutral_charges = [q + correction for q in charges]
    else:
        neutral_charges = charges

    return neutral_charges


class LAD(object):
    """
    Class to define Local Atomic Descriptor (LAD) object and
    operations such as comparsion/overlap between two LAD instances.
    """

    def __init__(self, params, atomic_number, charge, mcn, name=None):
        """
        @type params: dict
        @param params: defines the flavor of the LAD, such as parameters used
                       to construct the MCN and overlap, and charge type.

        @type atomic_number: int
        @param atomic_number: atomic number of atom at centre of LAD.

        @type charge: int
        @param charge: atomic charge of atom at centre of LAD.

        @type mcn: MCN object
        @param mcn: Multidimensional Connectivity Number (MCN) of atom.

        @type name: str
        @param name: optional name for later reference
        """

        self._nested_mcn = params['nested_MCN']
        self._wm = params['wm']
        self._wq = params['wq']
        self._sigmoid_s = params['sigmoid_s']
        self._sigmoid_c = params['sigmoid_c']
        self._charge = charge
        self._atomic_number = atomic_number
        self._mcn = mcn
        self._name = name

    @property
    def atomic_number(self):
        return self._atomic_number

    @property
    def charge(self):
        return self._charge

    @property
    def mcn(self):
        return self._mcn

    def _sigmoid(self, x):
        """
        Weighting function to attenuate the LAD overlap L-values.
        This could be folded into the definition of "overlap" itself,
        but for historical reasons, it is a separate operation.
        """
        s = self._sigmoid_s
        c = self._sigmoid_c
        return 1.0 / (np.exp(s * (c - x)) + 1.0)

    def overlap(self, other):
        """
        Compute overlap (L-value) between two LAD instances.

        @type other: LAD object
        @param other: another LAD instance, must be compatible with self.
                      e.g. both self and other must have nested-MCN, or not.

        @raise TypeError if LAD instances are incompatible.

        @return: (weight, L-value)
        """

        if not isinstance(other, LAD):
            msg = 'must be ' + str(type(self)) + ', not ' + str(type(other))
            raise TypeError(msg)

        if self._wm != other._wm or \
           self._wq != other._wq:
            msg = 'LAD instances have different distance metrics.'
            raise TypeError(msg)

        if self._nested_mcn != other._nested_mcn:
            msg = 'Both LAD instances must use nested-MCN, or not.'
            raise TypeError(msg)

        # Loop over all elements
        if self.atomic_number == other.atomic_number:
            # MCN delta
            mcn_delta = 0.0
            elements = set(self.mcn.keys())
            elements.update(other.mcn.keys())
            for e in elements:
                if self._nested_mcn:
                    # Nested-MCN
                    s_mcn = self.mcn.get(e, [])
                    o_mcn = other.mcn.get(e, [])
                    tmp = 0.0
                    for s, o in zip_longest(s_mcn, o_mcn, fillvalue=0.0):
                        tmp += abs(s - o)
                    mcn_delta += tmp**2
                else:
                    # Compact-MCN
                    mcn_delta += (
                        self.mcn.get(e, 0.0) - other.mcn.get(e, 0.0))**2
            # Charge delta
            charge_delta = (self.charge - other.charge)**2
            Lvalue = np.exp(-self._wm * mcn_delta - self._wq * charge_delta)
            return self._sigmoid(Lvalue), Lvalue

        else:
            return 0.0, 0.0


class MCN(object):
    """
    Container class for a Multidimensional Connectivity Number (MCN) for
    a single structure.  The MCN is available via lazy evaluation for any
    atom in the structure.
    """

    # Radii taken from DFTD3 code of S. Grimme which are in turn taken from
    # [P. Pyykko and M. Atsumi, Chem. Eur. J., 15, 186 (2009)]
    # Values for metals are decreased by 10% (following S. Grimme's
    # recommendadtion; actually the values below are copied from his
    # DFTD3 code, and copied here from /jaguar-src/main/aposteri/params.c
    # Units are Angstroms.
    #
    RADII = {
        1: 0.32,
        2: 0.46,
        3: 1.2,
        4: 0.94,
        5: 0.77,
        6: 0.75,
        7: 0.71,
        8: 0.63,
        9: 0.64,
        10: 0.67,
        11: 1.4,
        12: 1.25,
        13: 1.13,
        14: 1.04,
        15: 1.1,
        16: 1.02,
        17: 0.99,
        18: 0.96,
        19: 1.76,
        20: 1.54,
        21: 1.33,
        22: 1.22,
        23: 1.21,
        24: 1.1,
        25: 1.07,
        26: 1.04,
        27: 1.0,
        28: 0.99,
        29: 1.01,
        30: 1.09,
        31: 1.12,
        32: 1.09,
        33: 1.15,
        34: 1.1,
        35: 1.14,
        36: 1.17,
        37: 1.89,
        38: 1.67,
        39: 1.47,
        40: 1.39,
        41: 1.32,
        42: 1.24,
        43: 1.15,
        44: 1.13,
        45: 1.13,
        46: 1.08,
        47: 1.15,
        48: 1.23,
        49: 1.28,
        50: 1.26,
        51: 1.26,
        52: 1.23,
        53: 1.32,
        54: 1.31,
        55: 2.09,
        56: 1.76,
        57: 1.62,
        58: 1.47,
        59: 1.58,
        60: 1.57,
        61: 1.56,
        62: 1.55,
        63: 1.51,
        64: 1.52,
        65: 1.51,
        66: 1.5,
        67: 1.49,
        68: 1.49,
        69: 1.48,
        70: 1.53,
        71: 1.46,
        72: 1.37,
        73: 1.31,
        74: 1.23,
        75: 1.18,
        76: 1.16,
        77: 1.11,
        78: 1.12,
        79: 1.13,
        80: 1.32,
        81: 1.3,
        82: 1.3,
        83: 1.36,
        84: 1.31,
        85: 1.38,
        86: 1.42,
        87: 2.01,
        88: 1.81,
        89: 1.67,
        90: 1.58,
        91: 1.52,
        92: 1.53,
        93: 1.54,
        94: 1.55
    }

    def __init__(self, params, atno, carts):
        """
        :type params: dict
        :param params: parameters to define the flavor of MCN.

        :type  atno: list of ints
        :param atno: atomic number of each atom
        :type carts: numpy array
        :param carts: natom x 3 numpy array of cartesian coordinates
        """

        self._k1 = params['k1']
        self._k2 = params['k2']
        self._nested = params['nested_MCN']
        self._atno = list(atno)
        self._X = carts.copy()
        self._natoms = len(self._atno)
        self._mcn = OrderedDict()
        self._compressed = True

        assert len(self._atno) == len(self._X)

    def __getitem__(self, idx):
        """
        @type idx: integer
        @param idx: atom index in structure to return MCN

        @return: MCN instance for one atom
        """

        if idx not in self._mcn:
            self._build_mcn_for_atom(idx)
        return self._mcn[idx]

    def _build_mcn_for_atom(self, idx):
        """
        @type idx: integer
        @param idx: atom index in to build MCN (starts at zero)
        """

        mcn = defaultdict(list)
        radi = self.RADII[self._atno[idx]]
        for j in range(self._natoms):
            if j != idx:
                radj = self.RADII[self._atno[j]]
                rij = self._distance(idx, j)
                zij = self._k1 * self._k2 * (radi + radj) / rij
                tmp = 1.0 / (1.0 + np.exp(self._k1 - zij))
                # use string for json compatibility
                element = str(self._atno[j])
                mcn[element].append(tmp)

        for element in mcn:
            if self._nested:
                mcn[element] = sorted(mcn[element], reverse=True)
            else:
                mcn[element] = sum(mcn[element])

        # Sort by element atomic number
        self._mcn[idx] = OrderedDict(sorted(mcn.items()))

    def _distance(self, i, j):
        """
        distance between atoms i and j
        """
        dx = self._X[i] - self._X[j] 
        return np.sqrt(np.dot(dx, dx))

class ReferenceLadData(object):
    """
    class that optimizes lad evaluation
    """
    def __init__(self, reference_data):
        """
        setup data 
        stores a dict which relates atomiic number to lad data
        where lad data is reference values and MCN
        these are stored as numpy arrays
        """
        self._atnos = sorted(reference_data.keys())
        self._ntypes = len(self._atnos)
        self._ref_data = {}
        for atno in self._atnos:
            ndata = len(reference_data[atno])
            values = np.zeros(ndata)
            mcns = np.zeros([ndata, self._ntypes])
            for idx, (v, lad_instance) in enumerate(reference_data[atno]):
                values[idx] = v  
                mcns[idx, :] = self._extract_mcn(lad_instance)

            self._ref_data[atno] = (values, mcns)

    def _extract_mcn(self, lad):
        values = np.zeros(self._ntypes)
        for j, e in enumerate(self._atnos):
            values[j] = lad.mcn.get(str(e), 0.0)
        return values

    def _old_weights(self, lad):
        
        weights, lvalues = [], []
        norm = 0.0
        value = 0.0
        for v, ref_lad in self._reference_lads[lad.atomic_number]:
            w, l = lad.overlap(ref_lad)
            norm += w
            value += w*v
            weights.append(w)
            lvalues.append(l)
    
        if norm > ZERO_NORM:
            value = value / norm

        return value, np.array(weights), np.array(lvalues)

    def interpolate(self, lad):
        """
        use LAD params to interpolate ref data to an input lad
        """
        atno = lad.atomic_number

        if atno not in self._ref_data:
            return 0.0

        mcn_data = self._extract_mcn(lad)
        values, mcn_refs = self._ref_data[atno]

        # mcn delta as a matrix
        delta = mcn_refs - mcn_data

        # sum the squares over each example
        delta2 = np.multiply(delta, delta)
        summed = np.sum(delta2, axis=1)

        # compute weights
        lval = np.exp(-lad._wm * summed)
        s = lad._sigmoid_s
        c = lad._sigmoid_c
        denom = np.exp(s * (c - lval)) + 1.0
        weights = 1.0 / denom
        norm = np.sum(weights)

        # average
        value = np.dot(values, weights) / norm

        return value
