import math
import numpy as np


class RawDataset():

    def __init__(self, all_Xs, all_ys=None):

        last_idx = 0
        offsets = []
        for x in all_Xs:
            new_idx = last_idx + x.shape[0]
            offsets.append((last_idx, new_idx))
            last_idx = new_idx

        self.all_ys = all_ys
        self.all_Xs = np.concatenate(all_Xs, axis=0)
        self.all_offsets = np.array(offsets, dtype=np.int32)

    def iterate(self, batch_size):

        n_batches = math.ceil(len(self.all_offsets) / batch_size)

        for batch_idx in range(n_batches):
            s_m_idx = batch_idx * batch_size
            e_m_idx = min((batch_idx+1) * batch_size, len(self.all_offsets))

            X_batch_offsets = self.all_offsets[s_m_idx:e_m_idx, :] # needs to be shrunk down to 
            # print(X_batch_offsets)

            s_a_idx = X_batch_offsets[0][0]
            e_a_idx = X_batch_offsets[-1][-1]

            # offsets needs to be modified into unity
            Xs = self.all_Xs[s_a_idx:e_a_idx, :]

            X_batch_offsets -= X_batch_offsets[0][0] # convert into batch-wise indices


            if self.all_ys is not None:
                yield Xs, X_batch_offsets, self.all_ys[s_m_idx:e_m_idx]
            else:
                yield Xs, X_batch_offsets, None

