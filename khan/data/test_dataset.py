import unittest

import numpy as np

from khan.data.dataset import RawDataset

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

        for idx, (_, _, ys) in enumerate(rdy.iterate(batch_size=3)):
            if idx == 0:
                np.testing.assert_array_equal(ys, np.arange(0, 3, dtype=np.float32))
            elif idx == 1:
                np.testing.assert_array_equal(ys, np.arange(3, 6, dtype=np.float32))
            elif idx == 2:
                np.testing.assert_array_equal(ys, np.arange(6, 8, dtype=np.float32))
            else:
                assert 0
