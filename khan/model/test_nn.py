import unittest
import time

import numpy as np
import tensorflow as tf

from khan.model.nn import AtomNN, MoleculeNN
from khan.utils.helpers import inv

from concurrent.futures import ThreadPoolExecutor

class TestNN(unittest.TestCase):

    def setUp(self):
        self.sess = tf.Session()

    def tearDown(self):
        self.sess.close()

    def test_atom_energy(self):

        ph = tf.placeholder(dtype=tf.float32)
        ann = AtomNN(ph, (32,16,1), "H")
        nrg_op = ann.atom_energies()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(nrg_op, feed_dict={
            ph: np.random.rand(7, 32),
        })

    def test_mol_energy(self):

        atom_types = np.array([
            0,
            2,
            0,
            1,
            2,
            0,
            0,
            0,
            0,
            1,
            2,
        ], dtype=np.int32)

        atom_feats = np.array([
            [1.0, 2.0, 3.0, 1e-4], # H
            [2.0, 1.0, 4.0, 1e-5], # N
            [0.5, 1.2, 2.3, 2e-3], # H
            [0.3, 1.7, 3.2, 1e-4], # C
            [0.6, 1.2, 1.1, 5e-3], # N
            [14.0, 23.0, 15.0, 2e-4], # H
            [2.0, 0.5, 0, 4], # H
            [2.3, 0.2, 0.4, 1e-4], # H

            [2.3, 0.2, 0.4, 1e-5], # H
            [0.3, 1.7, 3.2, 5e-4], # C
            [0.6, 1.2, 1.1, 8e-4], # N
        ], dtype=np.float32)

        offsets = np.array([(0,8), (8,11)], dtype=np.int32)

        aaf = [
            tf.placeholder(tf.float32), # 0
            tf.placeholder(tf.float32), # 1
            tf.placeholder(tf.float32), # 2
            tf.placeholder(tf.float32)  # 3
        ]
        to = tf.placeholder(tf.int32)
        gi = tf.placeholder(tf.int32)

        perm = atom_types.argsort()
        atom_feats_sorted = atom_feats[perm]

        atom_type_features = [
           np.array([
            [1.0, 2.0, 3.0, 1e-4], # H
            [0.5, 1.2, 2.3, 2e-3], # H
            [14.0, 23.0, 15.0, 2e-4], # H
            [2.0, 0.5, 0, 4], # H
            [2.3, 0.2, 0.4, 1e-4], # H
            [2.3, 0.2, 0.4, 1e-5]], dtype=np.float32), # H
           np.array([
            [0.3, 1.7, 3.2, 1e-4],
            [0.3, 1.7, 3.2, 5e-4]], dtype=np.float32), # C
           np.array([
            [2.0, 1.0, 4.0, 1e-5], # N
            [0.6, 1.2, 1.1, 5e-3], # N
            [0.6, 1.2, 1.1, 8e-4]], dtype=np.float32),
           np.zeros([0, 4], dtype=np.float32)
        ]

        gather_idxs = inv(perm) # [0, 8, 1, 6, 9, 2, 3, 4]
        np.testing.assert_array_equal(gather_idxs, [0, 8, 1, 6, 9, 2, 3, 4, 5, 7, 10])

        mi = tf.placeholder(tf.int32)
        mnn = MoleculeNN(
            type_map=["H", "C", "N", "O"],
            atom_type_features=aaf,
            # type_offsets=to,
            gather_idxs=gi,
            mol_idxs=mi,
            layer_sizes=(4, 4, 1))

        self.sess.run(tf.global_variables_initializer())

        mol_idxs = np.array([0,0,0,0,0,0,0,0,1,1,1], dtype=np.int32)

        mol_energies = self.sess.run(mnn.molecule_energies(), feed_dict={
            aaf[0]: atom_type_features[0],
            aaf[1]: atom_type_features[1],
            aaf[2]: atom_type_features[2],
            aaf[3]: atom_type_features[3],
            mi: mol_idxs,
            gi: gather_idxs,
        })

        assert mol_energies.shape == (2, )

        expected_energies = []

        for a_idx, a_feats in enumerate(atom_feats):
            ann = mnn.anns[atom_types[a_idx]]
            res = self.sess.run(ann.atom_energies(),
                feed_dict={
                    ann.features: np.expand_dims(a_feats, axis=0)
                }
            )
            expected_energies.append(res[0])

        expected_energies = np.array(expected_energies)

        np.testing.assert_almost_equal(mol_energies[0], np.sum(expected_energies[0:8]), decimal=6)
        np.testing.assert_almost_equal(mol_energies[1], np.sum(expected_energies[8:]), decimal=6)
