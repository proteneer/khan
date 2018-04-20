import unittest
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import numpy as np

ani_mod = tf.load_op_library('ani.so');
# sort_lib = tf.load_op_library('ani_sort.so');

from tensorflow.python import debug as tf_debug


def linearize(i, j, k, l):
    if j < i:
        tmp = i
        i = j
        j = tmp

    N = 4
    K = 8
    L = 4

    basis = (N*(N-1)//2 - (N-i) * (N-i-1)//2 +j)
    
    idx = basis*K*L + k*L + l


    return idx



def fC(r_ij, r_C):

    if r_ij <= r_C:
        return 0.5 * math.cos(math.pi*r_ij/r_C) + 0.5
    else:
        return 0.0


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

class TestFeaturizer(unittest.TestCase):

    def setUp(self):

        cp = tf.ConfigProto(log_device_placement=True, allow_soft_placement=False, device_count = {'GPU': 1})

        self.sess = tf.Session(config=cp)

    def tearDown(self):

        self.sess.close()

    def reference_feats(self, mol):

        R_Rs = [
            5.0000000e-01,
            7.5625000e-01,
            1.0125000e+00,
            1.2687500e+00,
            1.5250000e+00,
            1.7812500e+00,
            2.0375000e+00,
            2.2937500e+00,
            2.5500000e+00,
            2.8062500e+00,
            3.0625000e+00,
            3.3187500e+00,
            3.5750000e+00,
            3.8312500e+00,
            4.0875000e+00,
            4.3437500e+00
        ]

        R_Rc = 4.6
        R_eta = 16

        A_thetas = [
            0.0000000e+00,
            7.8539816e-01,
            1.5707963e+00,
            2.3561945e+00,
            3.1415927e+00,
            3.9269908e+00,
            4.7123890e+00,
            5.4977871e+00
        ]

        A_Rs = [
            5.0000000e-01,
            1.1500000e+00,
            1.8000000e+00,
            2.4500000e+00,
        ]

        A_Rc = 3.1;
        A_eta = 6.0;
        A_zeta = 8.0;

        all_feats = []

        for i_idx, row_i in enumerate(mol):

            radial_feats = np.zeros(len(R_Rs) * 4, dtype=np.float32)
            angular_feats = np.zeros(len(A_Rs) * len(A_thetas) * (4 * (4+1) // 2), dtype=np.float32)
            # print(angular_feats.shape)

            i_xyz = row_i[1:]
            i_type = int(row_i[0])

            for j_idx, row_j in enumerate(mol):

                if i_idx == j_idx:
                    continue

                j_xyz = row_j[1:]
                j_type = int(row_j[0])
                r_ij = np.linalg.norm(i_xyz - j_xyz)

                for r_idx, r_s in enumerate(R_Rs):
                    summand = fC(r_ij, R_Rc) * math.exp(-R_eta*math.pow(r_ij-r_s, 2))
                    radial_feats[j_type*len(R_Rs)+r_idx] += summand


                for k_idx, row_k in enumerate(mol):

                    if i_idx == k_idx:
                        continue

                    if j_idx >= k_idx:
                        continue

                    k_xyz = row_k[1:]
                    k_type = int(row_k[0])
                    r_ik = np.linalg.norm(i_xyz - k_xyz)
                    v_ij = i_xyz - j_xyz
                    v_ik = i_xyz - k_xyz

                    theta_ijk = angle(v_ij, v_ik)

                    if r_ij < A_Rc and r_ik < A_Rc:

                        for at_idx, at_t in enumerate(A_thetas):
                            for ar_idx, ar_s in enumerate(A_Rs):

                                fC_ij = fC(r_ij, A_Rc)
                                fC_ik = fC(r_ik, A_Rc)


                                summand = math.pow(1+math.cos(theta_ijk - at_t), A_zeta) * math.exp(-A_eta*math.pow((r_ij+r_ik)/2 - ar_s, 2)) * fC_ij * fC_ik
                                summand = pow(2, 1 - A_zeta) * summand

                                # print(j_idx, k_idx, at_idx, ar_idx)
                                # print("i,j,k,t,s", i_idx, j_idx, k_idx, at_idx, ar_idx, linearize(j_type, k_type, at_idx, ar_idx))
                                angular_feats[linearize(j_type, k_type, at_idx, ar_idx)] += summand


            all_feats.append(np.concatenate([radial_feats, angular_feats]))

        return np.array(all_feats)



    def test_featurizer(self):


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


        mol_idxs = np.array([0,0,0,0,0,0,0,0,1,1,1], dtype=np.int32)

        atom_types = atom_matrix[:, 0]
        x = atom_matrix[:, 1]
        y = atom_matrix[:, 2]
        z = atom_matrix[:, 3]

        ph_atom_types = tf.placeholder(dtype=np.int32)
        ph_xs = tf.placeholder(dtype=np.float32)
        ph_ys = tf.placeholder(dtype=np.float32)
        ph_zs = tf.placeholder(dtype=np.float32)
        ph_mol_idxs = tf.placeholder(dtype=np.int32)

        scatter_idxs, gather_idxs, atom_counts = ani_mod.ani_sort(ph_atom_types)

        mol_atom_counts = tf.segment_sum(tf.ones_like(ph_mol_idxs), ph_mol_idxs)
        mol_offsets = tf.cumsum(mol_atom_counts, exclusive=True)

        obtained_si, obtained_gi, obtained_ac = self.sess.run(
            [scatter_idxs, gather_idxs, atom_counts],
            feed_dict={
                ph_mol_idxs: mol_idxs,
                ph_atom_types: atom_types,
            })

        np.testing.assert_array_equal(obtained_ac, [6,2,3,0])
        np.testing.assert_array_equal(obtained_si, [0,0,1,0,1,2,3,4,5,1,2])
        np.testing.assert_array_equal(obtained_gi, [0,8,1,6,9,2,3,4,5,7,10])

        f0, f1, f2, f3 = ani_mod.featurize(
            ph_xs,
            ph_ys,
            ph_zs,
            ph_atom_types,
            mol_offsets,
            mol_atom_counts,
            scatter_idxs,
            atom_counts
        )

        f0, f1, f2, f3 = tf.reshape(f0, (-1, 384)), tf.reshape(f1, (-1, 384)), tf.reshape(f2, (-1, 384)), tf.reshape(f3, (-1, 384))
        scattered_features = tf.concat([f0, f1, f2, f3], axis=0)
        features = tf.gather(scattered_features, gather_idxs)
            
        obtained_features = self.sess.run(features, feed_dict={
            ph_xs: x,
            ph_ys: y,
            ph_zs: z,
            ph_mol_idxs: mol_idxs,
            ph_atom_types: atom_types
        })

        expected_features_mol1 = self.reference_feats(atom_matrix[:8, :])
        expected_features_mol2 = self.reference_feats(atom_matrix[8:, :])

        # radial components
        np.testing.assert_almost_equal(obtained_features[:8, :64], expected_features_mol1[:, :64], decimal=6)
        np.testing.assert_almost_equal(obtained_features[8:, :64], expected_features_mol2[:, :64], decimal=6)

        # angular components
        np.testing.assert_almost_equal(obtained_features[:8, 64:], expected_features_mol1[:, 64:], decimal=6)
        np.testing.assert_almost_equal(obtained_features[8:, 64:], expected_features_mol2[:, 64:], decimal=6)


if __name__ == "__main__":
    unittest.main()