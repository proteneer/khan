import unittest

import numpy as np
import tensorflow as tf

from khan.model.symmetrizer import Symmetrizer
from khan.model.ani import ANI

class TestANI(unittest.TestCase):

    def setUp(self):
        self.sess = tf.Session()

    def tearDown(self):
        self.sess.close()

    def test_end_to_end_no_queue(self):

        sym = Symmetrizer()
        layer_sizes = [sym.feature_size(), 64, 8, 1]
        atm = ["H", "C", "N", "O"]
        bam = tf.placeholder(dtype=tf.float32)
        mo = tf.placeholder(dtype=tf.int32)
        mi = tf.placeholder(dtype=tf.int32)
        gi = tf.placeholder(dtype=tf.int32)

        ani = ANI(atm, sym, bam, mo, mi, gi, layer_sizes)

        self.sess.run(tf.global_variables_initializer())

        atom_matrix = np.array([
            [0, 1.0, 2.0, 3.0], # H
            [2, 2.0, 1.0, 4.0], # N
            [0, 0.5, 1.2, 2.3], # H
            [1, 0.3, 1.7, 3.2], # C
            [2, 0.6, 1.2, 1.1], # N
            [0, 14.0, 23.0, 15.0], # H
            [0, 2.0, 0.5, 0.3], # H
            [0, 2.3, 0.2, 0.4], # H

            [0, 2.3, 0.2, 0.4], # H
            [1, 0.3, 1.7, 3.2], # C
            [2, 0.6, 1.2, 1.1]], dtype=np.float32)

        mol_offsets = np.array([(0, 8), (8, 11)], dtype=np.int32)
        gather_idxs = np.array([0, 8, 1, 6, 9, 2, 3, 4, 5, 7, 10])
        mol_idxs = np.array([0,0,0,0,0,0,0,0,1,1,1], dtype=np.int32)

        # test training
        self.sess.run(ani.predict_op(), feed_dict={
            ani.bam: atom_matrix,
            ani.mol_offsets: mol_offsets,
            ani.gather_idxs: gather_idxs,
            ani.mol_idxs: mol_idxs,
        })

        # test getting features
        atom_type_features = self.sess.run(ani.atom_type_feats, feed_dict={
            ani.bam: atom_matrix,
            ani.mol_offsets: mol_offsets
        })

        # test training
        self.sess.run(ani.predict_op(), feed_dict={
            ani.atom_type_feats[0]: atom_type_features[0],
            ani.atom_type_feats[1]: atom_type_features[1],
            ani.atom_type_feats[2]: atom_type_features[2],
            ani.atom_type_feats[3]: atom_type_features[3],
            ani.gather_idxs: gather_idxs,
            ani.mol_idxs: mol_idxs,
        })

        # test loss equivalence:
        l0 = self.sess.run(ani.mol_nrgs, feed_dict={
            ani.bam: atom_matrix,
            ani.mol_offsets: mol_offsets,
            ani.gather_idxs: gather_idxs,
            ani.mol_idxs: mol_idxs,
        })

        l1 = self.sess.run(ani.mol_nrgs, feed_dict={
            ani.atom_type_feats[0]: atom_type_features[0],
            ani.atom_type_feats[1]: atom_type_features[1],
            ani.atom_type_feats[2]: atom_type_features[2],
            ani.atom_type_feats[3]: atom_type_features[3],
            ani.gather_idxs: gather_idxs,
            ani.mol_idxs: mol_idxs
        })

        np.testing.assert_almost_equal(l0, l1)