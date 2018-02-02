import math
import glob
import numpy as np
import queue
import os

from multiprocessing.dummy import Pool
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

    def iterate(self, batch_size):

        n_batches = self.num_batches(batch_size)

        for batch_idx in range(n_batches):
            s_m_idx = batch_idx * batch_size
            e_m_idx = min((batch_idx+1) * batch_size, len(self.all_offsets))

            X_batch_offsets = self.all_offsets[s_m_idx:e_m_idx, :] # needs to be shrunk down to 

            s_a_idx = X_batch_offsets[0][0]
            e_a_idx = X_batch_offsets[-1][-1]

            Xs = self.all_Xs[s_a_idx:e_a_idx, :]

            X_batch_offsets -= X_batch_offsets[0][0] # convert into batch-wise indices

            if self.all_ys is not None:
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

        with tf.Session() as session:

            all_ais = [] # batches of atom type offsets
            all_xos = [] # batches of mol offsets lists
            all_ys = [] 

            def submitter():
                for b_idx, (x_b, x_o, yy) in enumerate(self.iterate(batch_size)):
                    session.run(put_op, feed_dict={
                        x_b_enq: x_b,
                        x_o_enq: x_o
                    })
                    all_ais.append(x_b[:, 0])
                    all_xos.append(x_o)
                    all_ys.append(yy)

            q = queue.Queue()

            def writer():
                fd = FeaturizedDataset(data_dir)
                for s_idx in range(self.num_batches(batch_size)):
                    print("writing...", s_idx)
                    feat_data = q.get()
                    ai, xo, xy = all_ais[s_idx], all_xos[s_idx], all_ys[s_idx]
                    all_ais[s_idx] = None
                    all_xos[s_idx] = None # clear memory
                    all_ys[s_idx] = None # clear memory

                    fd.write(s_idx, feat_data, ai, xo, xy)
                return fd

            executor = ThreadPoolExecutor(2)
            submit_future = executor.submit(submitter)
            write_future = executor.submit(writer)

            for _ in range(self.num_batches(batch_size)):
                q.put(session.run(feat_op))

            submit_future.result()
            fd = write_future.result()

            return fd

def generate_fnames(data_dir, s_idx):
    return [
        os.path.join(data_dir, str(s_idx)+"_0.npy"),
        os.path.join(data_dir, str(s_idx)+"_1.npy"),
        os.path.join(data_dir, str(s_idx)+"_2.npy"),
        os.path.join(data_dir, str(s_idx)+"_3.npy"),
        os.path.join(data_dir, str(s_idx)+"_gi.npy"),
        os.path.join(data_dir, str(s_idx)+"_mi.npy"),
        os.path.join(data_dir, str(s_idx)+"_ys.npy")
    ]    

class FeaturizedDataset():

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def iterate(self, shuffle=False):
        n_shards = self.num_batches()

        try:
            perm = np.arange(n_shards)

            if shuffle:
                np.random.shuffle(perm)

            def load_shard(s_idx):
                fnames = generate_fnames(self.data_dir, s_idx)
                res = []
                for f in fnames:
                    # print("loading", f)
                    res.append(np.load(f, allow_pickle=False, mmap_mode='r'))

                return res

            pool = Pool(1)  # mp.dummy aliases ThreadPool to Pool
            next_shard = pool.apply_async(load_shard, (perm[0],))

            for ss_idx, shard_idx in enumerate(perm):

                res = next_shard.get()
                if ss_idx != len(perm) - 1:
                    next_shard = pool.apply_async(load_shard, (shard_idx,))
                else:
                    pool.close()

                yield res

        except Exception as e:
            print("WTF OM?", e)

    def num_batches(self):
        try:
            return self._nb
        except AttributeError:
            path = os.path.join(self.data_dir, '*.npy')
            self._nb = len(glob.glob(os.path.join(self.data_dir, '*.npy'))) // 7
            return self._nb
        
    def write(self, s_idx, batched_feat_Xs, atom_type_idxs, mol_offsets, mol_ys):

        try:
            # save as a more efficient format for IO when training
            scatter_idxs = np.argsort(atom_type_idxs) # note that this isn't stable.
            scatter_atom_feats = batched_feat_Xs[scatter_idxs]


            atom_type_idxs = atom_type_idxs[scatter_idxs]
            atom_type_offsets = []

            last_idx = 0
            for a_idx, a_type in enumerate(atom_type_idxs):
                if a_idx == len(atom_type_idxs) - 1:
                    atom_type_offsets.append((last_idx, len(atom_type_idxs)))
                elif atom_type_idxs[a_idx] != atom_type_idxs[a_idx+1]:
                    atom_type_offsets.append((last_idx, a_idx))
                    last_idx = a_idx

            atom_type_offsets = np.array(atom_type_offsets, dtype=np.int32)

            gather_idxs = inv(scatter_idxs)
            mol_idxs = []

            for mol_idx, (s, e) in enumerate(mol_offsets):
                mol_idxs.append(np.ones(shape=(e-s,), dtype=np.int32)*mol_idx)

            mol_idxs = np.concatenate(mol_idxs)

            if mol_ys is None:
                mol_ys = np.zeros(0)

            feats = []
            for offsets in atom_type_offsets:
                start = offsets[0]
                end = offsets[1]
                feats.append(scatter_atom_feats[start:end])

            while len(feats) < 4:
                feats.append(np.zeros(shape=(0, feats[0].shape[1]), dtype=np.float32))

            objs = [feats[0], feats[1], feats[2], feats[3], gather_idxs, mol_idxs, mol_ys]
            fnames = generate_fnames(self.data_dir, s_idx)

            for o, f in zip(objs, fnames):
                np.save(f, o, allow_pickle=False)

        except Exception as e:
            print("??", e)