import unittest
import time

import numpy as np
import tensorflow as tf

from khan.model.nn import AtomNN, MoleculeNN
from concurrent.futures import ThreadPoolExecutor

class TestNN():

    def __init__(self):
        self.sess = tf.Session()

    def test_atom_nn(self):

        ph = tf.placeholder(dtype=tf.float32)
        ann = AtomNN(ph, (32,16,1), "H")
        nrg_op = ann.atom_energies()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(nrg_op, feed_dict={
            ph: np.random.rand(7, 32),
        })

    def test_molecule(self):

        atom_feats = np.array([
            [0, 1.0, 2.0, 3.0, 1e-4], # H
            [2, 2.0, 1.0, 4.0, 1e-5], # N
            [0, 0.5, 1.2, 2.3, 2e-3], # H
            [1, 0.3, 1.7, 3.2, 1e-4], # C
            [2, 0.6, 1.2, 1.1, 5e-3], # N
            [0, 14.0, 23.0, 15.0, 2e-4], # H
            [0, 2.0, 0.5, 0, 4], # H
            [0, 2.3, 0.2, 0.4, 1e-4], # H

            [0, 2.3, 0.2, 0.4, 1e-5], # H
            [1, 0.3, 1.7, 3.2, 5e-4], # C
            [2, 0.6, 1.2, 1.1, 8e-4]], dtype=np.float32)

        offsets = np.array([(0,8), (8,11)], dtype=np.int32)

        mm = tf.placeholder(tf.float32)
        oo = tf.placeholder(tf.int32)

        mnn = MoleculeNN(
            type_map=["H", "C", "N", "O"],
            batched_atom_features=mm,
            offsets=oo,
            layer_sizes=(4, 2, 1))


        self.sess.run(tf.global_variables_initializer())

        mol_energies = self.sess.run(mnn.molecule_energies(), feed_dict={
            mm: atom_feats,
            oo: offsets,
            })

        assert mol_energies.shape == (2,)

        energies = []

        for a_idx, a_feats in enumerate(atom_feats):
            a_type = np.int32(a_feats[0])
            ann = mnn.anns[a_type]
            res = self.sess.run(ann.atom_energies(),
                feed_dict={
                    ann.features: np.expand_dims(a_feats[1:], axis=0)
                }
            )
            energies.append(res)

        energies = np.array(energies)

        np.testing.assert_almost_equal(mol_energies[0], np.sum(energies[0:8]))
        np.testing.assert_almost_equal(mol_energies[1], np.sum(energies[8:]))

    def test_benchmark(self):

        batch_size = 2048

        mc_enqueue = tf.placeholder(dtype=tf.float32)
        mi_enqueue = tf.placeholder(dtype=tf.int32)

        staging = tf.contrib.staging.StagingArea(capacity=10, dtypes=[tf.float32, tf.int32])

        put_op = staging.put([mc_enqueue, mi_enqueue])
        get_op = staging.get()

        mnn = MoleculeNN(
            type_map=["H", "C", "N", "O"],
            batched_atom_features=get_op[0],
            offsets=get_op[1],
            layer_sizes=(458, 256, 128, 64, 1))

        results_all = mnn.molecule_energies()

        self.sess.run(tf.global_variables_initializer())

        batches_per_thread = 256
        n_threads = 2

        def submitter():
            tot_time = 0
            # for i in range(batches_per_thread):

            mol_feats = []
            mol_offsets = []
            last_idx = len(mol_feats)

            for mol_idx in range(batch_size):
                # num_atoms = np.random.randint(12,30)
                num_atoms = np.random.randint(18)
                for i in range(num_atoms):

                    feats = np.random.rand(459)
                    feats[0] = atom_type = np.random.randint(0,4)

                    mol_feats.append(feats)

                mol_offsets.append((last_idx, len(mol_feats)))
                last_idx = len(mol_feats)

            mol_feats = np.array(mol_feats, dtype=np.float32)
            mol_offsets = np.array(mol_offsets, dtype=np.int32)

            for i in range(batches_per_thread):
                print("submitting", i)

                st = time.time()
                self.sess.run(put_op, feed_dict={
                    mc_enqueue: mol_feats,
                    mi_enqueue: mol_offsets,
                })

            return tot_time

        def runner():
            tot_time = 0
            for i in range(batches_per_thread):
                print("running", i)
                st = time.time()
                # this needs to transfer data back, but in practice we just compute
                # a loss and call it a day
                results = self.sess.run(results_all) 
                tot_time +=  time.time() - st

            return tot_time

        executor = ThreadPoolExecutor(8)

        for p in range(n_threads):
            executor.submit(submitter)

        futures = []
        delta = 0

        # run_executor = ThreadPoolExecutor(4)

        for p in range(n_threads):
            futures.append(executor.submit(runner))

        for f in futures:
            delta += f.result()

        tpm = delta/(batches_per_thread*n_threads*batch_size)
        print("Time Per Mol:", tpm, "seconds")
        print("Samples per minute:", 60/tpm)




    # def test_benchmark(self):

    #     batch_size = 1024

    #     queue = tf.InputFIFOQueue(capacity=5, dtypes=tf.float32)

    #     mc = tf.placeholder(tf.float32)
    #     mi = tf.placeholder(tf.int32)

    #     mnn = MoleculeNN(
    #         type_map=["H", "C", "N", "O"],
    #         batched_atom_features=mc,
    #         offsets=mi,
    #         layer_sizes=(458, 128, 128, 64, 1))

    #     results_all = mnn.molecule_energies()

    #     self.sess.run(tf.global_variables_initializer())

    #     batches_per_thread = 128
    #     n_threads = 4

    #     def closure():
    #         tot_time = 0
    #         # for i in range(batches_per_thread):

    #         mol_feats = []
    #         mol_offsets = []
    #         last_idx = len(mol_feats)

    #         for mol_idx in range(batch_size):
    #             # num_atoms = np.random.randint(12,30)
    #             num_atoms = np.random.randint(18)
    #             for i in range(num_atoms):

    #                 feats = np.random.rand(459)
    #                 feats[0] = atom_type = np.random.randint(0,4)

    #                 mol_feats.append(feats)

    #             mol_offsets.append((last_idx, len(mol_feats)))
    #             last_idx = len(mol_feats)

    #         mol_feats = np.array(mol_feats, dtype=np.float32)
    #         mol_offsets = np.array(mol_offsets, dtype=np.int32)

    #         for i in range(batches_per_thread):
    #             print("i", i)
    #             # mol_feats = []
    #             # mol_offsets = []
    #             # last_idx = len(mol_feats)

    #             st = time.time()
    #             combined = self.sess.run(results_all, feed_dict={
    #                 mc: mol_feats,
    #                 mi: mol_offsets
    #             })

    #             # if i > 0:
    #             tot_time += time.time() - st

    #         return tot_time

    #     executor = ThreadPoolExecutor(4)
    #     futures = []
    #     delta = 0

    #     for p in range(n_threads):
    #         futures.append(executor.submit(closure))
    #     for f in futures:
    #         delta += f.result()

    #     tpm = delta/(batches_per_thread*n_threads*batch_size)
    #     print("Time Per Mol:", tpm, "seconds")
    #     print("Samples per minute:", 60/tpm)




