import unittest
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np

from khan.model import symmetrizer
from tensorflow.python import debug as tf_debug

class TestSymmetrizer(unittest.TestCase):

    def setUp(self):

        self.sess = tf.Session()

    def tearDown(self):

        self.sess.close()

    def test_radial_symmetry(self):

        atom_matrix = np.array([
            [0, 1.0, 2.0, 3.0], # H
            [2, 2.0, 1.0, 4.0], # N
            [0, 0.5, 1.2, 2.3], # H
            [1, 0.3, 1.7, 3.2], # C
            [2, 0.6, 1.2, 1.1], # N
            [0, 14.0, 23.0, 15.0], # H
        ], dtype=np.float32)

        coords_matrix = atom_matrix[:, 1:]

        R_Rs = (0.5, 0.7)
        R_Rc = 16
        R_eta = 0.3

        sym = symmetrizer.Symmetrizer(R_Rs=R_Rs, R_Rc=R_Rc, R_eta=R_eta)
        feat = sym.radial_symmetry(tf.constant(atom_matrix))

        result = self.sess.run(feat)

        assert result.shape == (atom_matrix.shape[0], len(sym.R_Rs)*sym.max_atom_types)

        # groupings
        # 0
        # 2   1
        # 5 3 4 0
        # H C N O

        # H block:
        #     0 2 5
        # 0.5 x   x
        # 0.7 x   x

        # test atom 0
        r_01 = np.linalg.norm(coords_matrix[0] - coords_matrix[2])
        r_05 = np.linalg.norm(coords_matrix[0] - coords_matrix[5]) # > Rc 16 cutoff

        fC = 0.5 * math.cos(math.pi * r_01/R_Rc) + 0.5
        expected_r_0 = math.exp(-R_eta*math.pow(r_01-R_Rs[0], 2)) * fC

        np.testing.assert_almost_equal(expected_r_0, 0.8607852536605668, decimal=6)
        np.testing.assert_almost_equal(result[0][0], expected_r_0, decimal=6)

        fC = 0.5 * math.cos(math.pi * r_01/R_Rc) + 0.5
        expected_r_1 = math.exp(-R_eta*math.pow(r_01-R_Rs[1], 2)) * fC

        np.testing.assert_almost_equal(result[0][1], expected_r_1, decimal=6)

        # test atom 1
        r_10 = np.linalg.norm(coords_matrix[1] - coords_matrix[0])
        r_11 = np.linalg.norm(coords_matrix[1] - coords_matrix[2])
        r_15 = np.linalg.norm(coords_matrix[1] - coords_matrix[5]) # > R_Rc 16 cutoff

        fC0 = 0.5 * math.cos(math.pi * r_10/R_Rc) + 0.5
        expected_r_0 = math.exp(-R_eta*math.pow(r_10-R_Rs[0], 2)) * fC0
        fC1 = 0.5 * math.cos(math.pi * r_11/R_Rc) + 0.5
        expected_r_1 = math.exp(-R_eta*math.pow(r_11-R_Rs[0], 2)) * fC1

        np.testing.assert_almost_equal(result[1][0], expected_r_0 + expected_r_1, decimal=6)

        # N block:
        #   1 4

        # test atom 3, second Rs
        r_31 = np.linalg.norm(coords_matrix[3] - coords_matrix[1])
        r_34 = np.linalg.norm(coords_matrix[3] - coords_matrix[4])

        fC_r31 = 0.5 * math.cos(math.pi * r_31/R_Rc) + 0.5
        expected_r_31 = math.exp(-R_eta*math.pow(r_31-R_Rs[1], 2)) * fC_r31
        fC_r34 = 0.5 * math.cos(math.pi * r_34/R_Rc) + 0.5
        expected_r_34 = math.exp(-R_eta*math.pow(r_34-R_Rs[1], 2)) * fC_r34

        # 012345 -> 5 corresponds to Rs1
        # HHCCNN
        np.testing.assert_almost_equal(result[3][5], expected_r_31 + expected_r_34, decimal=6)

    def test_angular_symmetry(self):

        atom_matrix = np.array([
            [0, 1.0, 2.0, 3.0], # H
            [2, 2.0, 1.0, 4.0], # N
            [0, 0.5, 1.2, 2.3], # H
            [1, 0.3, 1.7, 3.2], # C
            [2, 0.6, 1.2, 1.1], # N
            [0, 14.0, 23.0, 15.0], # H
            [0, 2.0, 0.5, 0.3], # H
            [0, 2.3, 0.2, 0.4], # H
        ], dtype=np.float32)

        coords_matrix = atom_matrix[:, 1:]

        A_Rs = (0.3, 0.5)
        A_Rc = 16
        A_eta = 0.3
        A_zeta = 0.01
        A_thetas = (0.1, 0.11)

        sym = symmetrizer.Symmetrizer(A_Rs=A_Rs, A_Rc=A_Rc, A_eta=A_eta, A_zeta=A_zeta, A_thetas=A_thetas)
        angular_feat = sym.angular_symmetry(tf.constant(atom_matrix))

        obtained = self.sess.run(angular_feat)

        assert np.isfinite(obtained).all()

        assert obtained.shape == (atom_matrix.shape[0], len(sym.A_Rs)*len(sym.A_thetas)*sym.max_atom_types*(sym.max_atom_types+1)/2)

        # compute 0-HH features

        # for atom 2, which is an H
        #                                   x  x
        # the HH feature should include (2[05,50,06,60,07,70]), but 5-x distance is outside of the shell. So the only contribution should be
        # from the 205. The interactions 2[56,65,57,75] should be excluded since it's outside of the distance cutoff.
        pair_lists = [(0, 6), (6, 0), (0, 7), (7, 0), (6, 7), (7, 6)]

        expected = np.zeros((2, 2))

        for r_idx, Rs in enumerate(A_Rs):
            for t_idx, theta in enumerate(A_thetas):
                cum_sum = 0
                for j, k in pair_lists:
                    v_ij = coords_matrix[2] - coords_matrix[j]
                    v_ik = coords_matrix[2] - coords_matrix[k]
                    R_ij = np.linalg.norm(v_ij)
                    R_ik = np.linalg.norm(v_ik)

                    fC_Rij = 0.5 * math.cos(math.pi * R_ij/A_Rc) + 0.5
                    fC_Rik = 0.5 * math.cos(math.pi * R_ik/A_Rc) + 0.5

                    rhs = np.exp(-A_eta*np.power((R_ij + R_ik)/2 - Rs, 2.0)) * fC_Rij * fC_Rik
                    theta_ijk = np.arccos(np.dot(v_ij, v_ik) / (R_ij * R_ik))
                    lhs = np.power(1 + np.cos(theta_ijk - theta), A_zeta)

                    cum_sum += lhs * rhs

                expected[r_idx][t_idx] = cum_sum * np.power(2.0, 1 - A_zeta)

        np.testing.assert_almost_equal(expected[0][0], obtained[2][0], decimal=6)
        np.testing.assert_almost_equal(expected[0][1], obtained[2][1], decimal=6)
        np.testing.assert_almost_equal(expected[1][0], obtained[2][2], decimal=6)
        np.testing.assert_almost_equal(expected[1][1], obtained[2][3], decimal=6)

    def test_featurize(self):

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

        sym = symmetrizer.Symmetrizer()
        # result = sym.radial_symmetry(tf.constant(atom_matrix[]))
        result_1 = sym.featurize_one(tf.constant(atom_matrix[:8, :], dtype=tf.float32))
        result_2 = sym.featurize_one(tf.constant(atom_matrix[8:, :], dtype=tf.float32))
        first = self.sess.run(result_1)
        second = self.sess.run(result_2)

        mol_offsets = np.array([(0, 8), (8, 11)], dtype=np.int32)

        mc = tf.placeholder(tf.float32)
        mo = tf.placeholder(tf.int32)

        results_all = sym.featurize_batch(mc, mo)
        combined = self.sess.run(results_all, feed_dict={
            mc: atom_matrix,
            mo: mol_offsets,
        })

        expected = np.concatenate([first, second], axis=0)

        np.testing.assert_array_almost_equal(expected, combined)

    def test_benchmark(self):

        num_mols_per_batch = 6
        sym = symmetrizer.Symmetrizer()

        mc = tf.placeholder(tf.float32)
        mi = tf.placeholder(tf.int32)

        results_all = sym.featurize_batch(mc, mi)

        batches_per_thread = 32
        n_threads = 2

        def closure():
            tot_time = 0
            for i in range(batches_per_thread):
                mol_coords = []
                mol_offsets = []
                last_idx = len(mol_coords)

                for mol_idx in range(num_mols_per_batch):
                    num_atoms = np.random.randint(12,64)
                    for i in range(num_atoms):
                        atom_type = np.random.randint(0,4)
                        x = np.random.rand()
                        y = np.random.rand()
                        z = np.random.rand()
                        mol_coords.append((atom_type, x, y, z))

                    mol_offsets.append((last_idx, len(mol_coords)))
                    last_idx = len(mol_coords)

                mol_coords = np.array(mol_coords, dtype=np.float32)
                mol_offsets = np.array(mol_offsets, dtype=np.int32)

                st = time.time()
                combined = self.sess.run(results_all, feed_dict={
                    mc: mol_coords,
                    mi: mol_offsets
                })

                if i > 0:
                    tot_time += time.time() - st

            return tot_time

        executor = ThreadPoolExecutor(4)
        futures = []
        delta = 0

        for p in range(n_threads):
            futures.append(executor.submit(closure))
        for f in futures:
            delta += f.result()

        print("Time Per Mol:", delta/((batches_per_thread-1)*n_threads*num_mols_per_batch))





        # num_mols_per_batch = 256


        # profiling/benchmarking
        # mol_coords = []
        # mol_offsets = []
        # last_idx = len(mol_coords)

        # for mol_idx in range(num_mols_per_batch):
        #     num_atoms = np.random.randint(12,64)
        #     for i in range(num_atoms):
        #         atom_type = np.random.randint(0,4)
        #         x = np.random.rand()
        #         y = np.random.rand()
        #         z = np.random.rand()
        #         mol_coords.append((atom_type, x, y, z))

        #     mol_offsets.append((last_idx, len(mol_coords)))
        #     last_idx = len(mol_coords)

        # mol_coords = np.array(mol_coords, dtype=np.float32)
        # mol_offsets = np.array(mol_offsets, dtype=np.int32)


        # dbg_sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        # combined = dbg_sess.run(results_all, feed_dict={mc: mol_coords, mi: mol_offsets})
        # dbg_sess.close()



        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        self.sess.run(results_all,
                feed_dict={
                            mc: mol_coords,
                            mi: mol_offsets,
                        },
                options=options,
                run_metadata=run_metadata)
        self.sess.close()

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('/home/yutong/while_loop2.json', 'w') as f:
            f.write(chrome_trace)

        # print("Time Per Mol:", (time.time() - st)/num_mols)

        # print(first, second, combined)
if __name__ == "__main__":
    unittest.main()