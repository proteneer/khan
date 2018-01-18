from tensorflow.python.client import timeline

from io import StringIO

import numpy as np
import time
import tensorflow as tf
import unittest

from khan.model.nn import MoleculeNN
from khan.training.trainer import Trainer

from concurrent.futures import ThreadPoolExecutor

# invert a permutation
def inv(p):
    inverse = [0] * len(p)
    for i, p in enumerate(p):
        inverse[p] = i
    return inverse

class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.sess = tf.Session()

    def tearDown(self):
        self.sess.close()

    def test_benchmark(self):

        batch_size = 1024

        f0_enq = tf.placeholder(dtype=tf.float32)
        f1_enq = tf.placeholder(dtype=tf.float32)
        f2_enq = tf.placeholder(dtype=tf.float32)
        f3_enq = tf.placeholder(dtype=tf.float32)
        gi_enq = tf.placeholder(dtype=tf.int32)
        mi_enq = tf.placeholder(dtype=tf.int32)
        yt_enq = tf.placeholder(dtype=tf.float32)

        staging = tf.contrib.staging.StagingArea(
            capacity=10, dtypes=[
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.int32,
                tf.int32,
                tf.float32])

        put_op = staging.put([f0_enq, f1_enq, f2_enq, f3_enq, gi_enq, mi_enq, yt_enq])
        get_op = staging.get()

        feat_size = 768

        f0, f1, f2, f3, gi, mi, yt = get_op[0], get_op[1], get_op[2], get_op[3], get_op[4], get_op[5], get_op[6]

        mnn = MoleculeNN(
            type_map=["H", "C", "N", "O"],
            atom_type_features=[f0, f1, f2, f3],
            gather_idxs=gi,
            mol_idxs=mi,
            layer_sizes=(feat_size, 256, 128, 64, 1))

        trainer = Trainer(mnn, yt)
        results_all = trainer.train()

        self.sess.run(tf.global_variables_initializer())

        batches_per_thread = 1024
        n_threads = 1

        def submitter():
            tot_time = 0

            mol_idxs = []
            mol_feats = []
            atom_types = []
            split_indices = [[], [], [] ,[]]

            global_idx = 0

            for mol_idx in range(batch_size):
                num_atoms = np.random.randint(16, 17)

                for i in range(num_atoms):
                    mol_feats.append(np.random.rand(feat_size))
                    mol_idxs.append(mol_idx)
                    a_type = np.random.randint(0,4)
                    atom_types.append(a_type)
                    split_indices[a_type].append(global_idx)
                    global_idx += 1

            mol_idxs = np.array(mol_idxs, dtype=np.int32)
            mol_feats = np.array(mol_feats, dtype=np.float32)
            atom_types = np.array(atom_types, dtype=np.int32)

            perm = atom_types.argsort()
            gather_idxs = inv(perm)

            t0, t1, t2, t3 = mol_feats[split_indices[0]], \
                mol_feats[split_indices[1]], \
                mol_feats[split_indices[2]], \
                mol_feats[split_indices[3]]

            mol_yy = np.random.rand(batch_size)

            for i in range(batches_per_thread):
                self.sess.run(put_op, feed_dict={
                    f0_enq: t0,
                    f1_enq: t1,
                    f2_enq: t2,
                    f3_enq: t3,
                    gi_enq: gather_idxs,
                    mi_enq: mol_idxs,
                    yt_enq: mol_yy,
                })

            return tot_time


        executor = ThreadPoolExecutor(4)

        for p in range(n_threads):
            executor.submit(submitter)

        tot_time = 0
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        for i in range(batches_per_thread):
            print("running", i)
            st = time.time()

            # this needs to transfer data back, but in practice we just compute
            # a loss and call it a day
            # if i < 10:
            if True:
                print("waiting...")
                self.sess.run(results_all)
            else:

                # this needs to transfer data back, but in practice we just compute
                # a loss and call it a day
                self.sess.run(results_all,
                    options=options,
                    run_metadata=run_metadata)

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('/home/yutong/train.json', 'w') as f:
                    f.write(chrome_trace)

                assert 0

            tot_time +=  time.time() - st # this logic is a little messed up

        tpm = tot_time/(batches_per_thread*n_threads*batch_size)
        print("Time Per Mol:", tpm, "seconds")
        print("Samples per minute:", 60/tpm)


        # s = StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

if __name__ == "__main__":
    unittest.main()