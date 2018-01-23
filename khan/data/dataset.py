import math
import glob
import numpy as np
import queue
import os

from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from khan.model.symmetrizer import Symmetrizer
from khan.utils.helpers import inv, compute_offsets

class RawDataset():

    def __init__(self, all_Xs, all_ys=None):

        offsets = compute_offsets(all_Xs)

        self.all_ys = all_ys
        self.all_Xs = np.concatenate(all_Xs, axis=0)
        self.all_offsets = np.array(offsets, dtype=np.int32)

    def num_mols(self):
        return len(self.all_offsets)

    def num_batches(self, batch_size):
        return math.ceil(self.num_mols() / batch_size)

    def iterate(self, batch_size, load_ys=False):

        n_batches = self.num_batches(batch_size)

        for batch_idx in range(n_batches):
            s_m_idx = batch_idx * batch_size
            e_m_idx = min((batch_idx+1) * batch_size, len(self.all_offsets))

            X_batch_offsets = self.all_offsets[s_m_idx:e_m_idx, :] # needs to be shrunk down to 

            s_a_idx = X_batch_offsets[0][0]
            e_a_idx = X_batch_offsets[-1][-1]

            Xs = self.all_Xs[s_a_idx:e_a_idx, :]

            X_batch_offsets -= X_batch_offsets[0][0] # convert into batch-wise indices

            if self.all_ys is not None and load_ys:
                yield Xs, X_batch_offsets, self.all_ys[s_m_idx:e_m_idx]
            else:
                yield Xs, X_batch_offsets, None

    def featurize(self, batch_size, data_dir, symmetrizer=None):

        print("featurizing", self.num_batches(batch_size), "batches")
        # multithreaded featurization code:

        # 1. Master thread executes session.run(feat_op) to get the featurized data
        # 2. Child thread #1 feeds into the StagingArea queue system
        # 3. Child thread #2 writes the featurized data to disk

        x_b_enq = tf.placeholder(dtype=tf.float32)
        x_o_enq = tf.placeholder(dtype=tf.int32)

        staging = tf.contrib.staging.StagingArea(
            capacity=10,
            dtypes=[tf.float32, tf.int32],
        )

        put_op = staging.put([x_b_enq, x_o_enq])
        get_op = staging.get()

        if symmetrizer is None:
            symmetrizer = Symmetrizer()
        
        feat_op = symmetrizer.featurize_batch(get_op[0], get_op[1])
        session = tf.Session()

        all_xos = [] # lists are thread safe due to GIL
        all_ys = [] # lists are thread safe

        def submitter():
            for b_idx, (x_b, x_o, yy) in enumerate(self.iterate(batch_size, load_ys=False)):
                session.run(put_op, feed_dict={
                    x_b_enq: x_b,
                    x_o_enq: x_o
                })
                all_xos.append(x_o)
                all_ys.append(yy)

        q = queue.Queue()

        def writer():
            fd = FeaturizedDataset(data_dir)
            for s_idx in range(self.num_batches(batch_size)):
                feat_data = q.get()
                xo, xy = all_xos[s_idx], all_ys[s_idx]
                all_xos[s_idx] = None # clear memory
                all_ys[s_idx] = None # clear memory
                fd.write(s_idx, feat_data, xo, xy)
            return fd

        executor = ThreadPoolExecutor(2)
        submit_future = executor.submit(submitter)
        write_future = executor.submit(writer)

        for _ in range(self.num_batches(batch_size)):
            q.put(session.run(feat_op))

        submit_future.result()
        fd = write_future.result()

        session.close()
        return fd


class FeaturizedDataset():

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def iterate(self, shuffle=False):
        path = os.path.join(self.data_dir, '*.npz')
        n_shards = len(glob.glob(os.path.join(self.data_dir, '*.npz')))

        perm = np.arange(n_shards)

        if shuffle:
            np.shuffle(perm)

        for shard_idx in perm:
            path = os.path.join(self.data_dir, str(shard_idx) + ".npz")
            res = np.load(path, allow_pickle=False)
            yield res['af'], res['gi'], res['mi']

    def write(self, s_idx, batched_Xs, mol_offsets, mol_ys):
        # save as a more efficient format for IO when training
        scatter_idxs = np.argsort(batched_Xs[:, 0],) # note that this isn't stable.
        atom_feats = batched_Xs[scatter_idxs]
        gather_idxs = inv(scatter_idxs)
        mol_idxs = []

        for mol_idx, (s, e) in enumerate(mol_offsets):
            mol_idxs.append(np.ones(shape=(e-s,), dtype=np.int32)*mol_idx)

        mol_idxs = np.concatenate(mol_idxs)

        fname = os.path.join(self.data_dir, str(s_idx)+".npz")

        with open(fname, "wb") as fh:
            np.savez(fh, af=atom_feats, gi=gather_idxs, mi=mol_idxs)
