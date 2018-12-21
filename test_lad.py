import unittest
import numpy as np
import os

from khan.lad import lad
from khan.lad import lad_coulomb
from khan.utils.helpers import atomic_number_to_atom_id
from khan.utils import constants
from khan.utils.constants import KCAL_MOL_IN_HARTREE

example_carts = [
[6,  0.000000, 0.000000, 0.000000,   -0.177224923127194],
[6,  1.500000, 0.000000, 0.000000,   -0.018485784983915],
[6,  2.000000, -1.414214, -0.000000,  0.515235757335987],
[8,  2.149740, -1.837729, 1.280786,  -0.636383407371812],
[8,  2.248335, -2.116612, -0.973473, -0.452251119362869],
[1,  -0.363333, 1.027662, 0.000000,   0.0719304893227678],
[1,  -0.363333, -0.513831, -0.889981, 0.0719305602025777],
[1,  -0.363333, -0.513831, 0.889981,  0.0719305561893141],
[1,  1.863333, 0.513831, 0.889981,    0.0719322668197482],
[1,  1.863333, 0.513831, -0.889981,   0.0719323474674757],
[1,  2.464227, -2.727234, 1.267070,   0.409453257507919]
]

class TestLad(unittest.TestCase):
    def test_interpolation(self):
        """
        Test that we can read in reference data
        and use it to interpolate charges
        """
        ref_data = os.path.join("data", "gdb6_reference_lad_data.json") 
        ref_lads = lad.read_reference_lads(ref_data, lad.LAD_PARAMS)

        atomic_numbers = np.array([atom[0] for atom in example_carts])
        carts = np.array([atom[1:4] for atom in example_carts])
        ref_q = np.array([atom[4] for atom in example_carts])

        test_q = lad.interpolate_charges(atomic_numbers, carts, lad.LAD_PARAMS, ref_lads)

        assert np.sum(abs(ref_q - test_q)) < 1.0e-6

        # test conversion from khan format as well
        conversion = {}
        X = np.array([[atomic_number_to_atom_id(atom[0]), atom[1], atom[2], atom[3]] for atom in example_carts])

        test_q = lad_coulomb.lad_charges(X, lad.LAD_PARAMS, ref_lads)
        assert np.sum(abs(ref_q - test_q)) < 1.0e-6

        
    def test_softmax(self):
        """
        Test three limits
        #1 goes to xcut when x << xcut
        #2 goes to x when x >> xcut
        #3 passes through x = xcut
        """

        value = lad_coulomb.softmax(2.0, 4.0, 10.0)
        assert abs(value-4.0) < 1.0e-6

        value = lad_coulomb.softmax(6.0, 4.0, 10.0)
        assert abs(value-6.0) < 1.0e-6

        value = lad_coulomb.softmax(4.0, 4.0, 10.0)
        assert abs(value-4.0) < 1.0e-6

    def test_pair_coulomb(self):
        """
        #1 pair of atoms that are far apart should be same as coulomb
        #2 at short range it should just be qiqi/4.0
        """

        X = np.array([
            [0, 4.0, 4.0, 4.0],
            [0, -1.0, -2.0, -3.0]
        ])
        q = np.array([0.25, -0.25])

        dX = np.sqrt(np.dot(X[0] - X[1], X[0] - X[1]))
        ref = q[0]*q[1]*constants.ANGSTROM_IN_BOHR/dX

        test = lad_coulomb.coulomb(X, q)

        assert abs(ref - test) < 1.0e-6

        X = np.array([
            [0, 0.0, 0.0, 0.0],
            [0, -1.0, 0.0, 0.0]
        ])
        test = lad_coulomb.coulomb(X, q)
        ref = q[0]*q[1]*constants.ANGSTROM_IN_BOHR/4.0
        
        assert abs(ref - test) < 1.0e-6

    def test_constants(self):
        assert abs(constants.ANGSTROM_IN_BOHR - 0.529) < 1.0e-3  
        assert abs(constants.KCAL_MOL_IN_HARTREE - 627.509) < 1.0e-3

if __name__ == "__main__":
    unittest.main()
