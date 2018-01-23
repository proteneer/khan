import unittest
import tempfile
import tensorflow as tf
import numpy as np

from khan.model.symmetrizer import Symmetrizer
from khan.data.dataset import RawDataset
from khan.utils.helpers import compute_offsets

class TestDataset():

    def test_raw_dataset_iterate(self):

        dummy_elem = [1.0, 2.0]

        all_Xs = [
            np.array([dummy_elem]*3), 
            np.array([dummy_elem]*4),
            np.array([dummy_elem]*1),
            np.array([dummy_elem]*2),
            np.array([dummy_elem]*8),
            np.array([dummy_elem]*9),
            np.array([dummy_elem]*3),
            np.array([dummy_elem]*5),
        ]

        rd = RawDataset(all_Xs)

        for idx, (x_batch, x_offsets, _) in enumerate(rd.iterate(batch_size=3)):
            if idx == 0:
                assert x_batch.shape[0] == 3+4+1
                np.testing.assert_array_equal(x_offsets, np.array([(0,3), (3,7), (7,8)], dtype=np.int32))
            elif idx == 1:
                x_batch.shape[0] == 2+8+9
                np.testing.assert_array_equal(x_offsets, np.array([(0,2), (2,10), (10,19)], dtype=np.int32))
            elif idx == 2:
                x_batch.shape[0] == 3+5
                np.testing.assert_array_equal(x_offsets, np.array([(0,3), (3,8)], dtype=np.int32))
            else:
                assert 0

        all_ys = np.arange(len(all_Xs), dtype=np.float32)

        rdy = RawDataset(all_Xs, all_ys)

        for idx, (_, _, ys) in enumerate(rdy.iterate(batch_size=3, load_ys=True)):
            if idx == 0:
                np.testing.assert_array_equal(ys, np.arange(0, 3, dtype=np.float32))
            elif idx == 1:
                np.testing.assert_array_equal(ys, np.arange(3, 6, dtype=np.float32))
            elif idx == 2:
                np.testing.assert_array_equal(ys, np.arange(6, 8, dtype=np.float32))
            else:
                assert 0

    def test_featurize(self):

        dummy_elem = [1.0, 2.0, 3.0, 4.0]
        all_Xs = []

        for mol_idx in range(125):
            num_atoms = np.random.randint(12,64)
            mol_coords = []
            for i in range(num_atoms):
                atom_type = np.random.randint(0, 4)
                x = np.random.rand()
                y = np.random.rand()
                z = np.random.rand()
                mol_coords.append((atom_type, x, y, z))
            all_Xs.append(np.array(mol_coords, dtype=np.float32))

        rd = RawDataset(all_Xs)

        sym = Symmetrizer()

        bam = tf.placeholder(tf.float32)
        bao = tf.placeholder(tf.int32)

        feat_op = sym.featurize_batch(bam, bao)

        with tf.Session() as sess:
            with tempfile.TemporaryDirectory() as tmpd:
                batch_size = 16
                fd = rd.featurize(batch_size=batch_size, data_dir=tmpd, symmetrizer=sym)
                for batch_idx, (af, gi, mi) in enumerate(fd.iterate(shuffle=False)):

                    results = af[gi]

                    s_m_idx = batch_idx*batch_size
                    e_m_idx = min((batch_idx+1)*batch_size, len(all_Xs))

                    pre_concat = all_Xs[s_m_idx:e_m_idx]

                    batch_Xs = np.concatenate(pre_concat, axis=0)
                    batch_offsets = np.array(compute_offsets(pre_concat), dtype=np.int32)

                    expected = sess.run(feat_op, feed_dict={
                        bam: batch_Xs,
                        bao: batch_offsets
                    })

                    np.testing.assert_array_equal(results, expected)

                    expected_mis = []
                    for m_idx, mm in enumerate(pre_concat):
                        expected_mis.extend([m_idx]*len(mm))

                    np.testing.assert_array_equal(mi, expected_mis)
