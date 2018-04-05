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
        """
        Construct a raw, unfeaturized dataset representing a collection of molecules.

        Params:
        -------

        all_Xs: list of np.array
            List of molecule coordinates, where each element in the list is an Nx3 numpy array.
        
        all_ys: np.array (optional)
            Numpy array representing the value of each molecule in the all_Xs array. This can be
            left blank if doing prediction.

        """
        offsets = compute_offsets(all_Xs)

        self.all_ys = all_ys
        self.all_Xs = all_Xs

    def num_mols(self):
        return len(self.all_Xs)

    def num_batches(self, batch_size):
        return math.ceil(self.num_mols() / batch_size)

    def iterate(self, batch_size, shuffle):

        perm = np.arange(len(self.all_Xs))
        if shuffle:
            np.random.shuffle(perm)

        n_batches = self.num_batches(batch_size)

        for batch_idx in range(n_batches):
            s_m_idx = batch_idx * batch_size
            e_m_idx = min((batch_idx+1) * batch_size, len(self.all_Xs))

            mol_Xs = []
            mol_ids = []

            for local_idx, p_idx in enumerate(perm[s_m_idx:e_m_idx]):
                # print("p_idx", p_idx)
                mol = self.all_Xs[p_idx]

                # do *not* remove this line. It's a super important sanity check since our
                # GPU kernels do not support larger than 32 atoms.
                if len(mol) > 32:
                    print("FATAL: Molecules with more than 32 atoms are not supported.")
                    assert 0

                mol_Xs.append(mol)
                mol_ids.extend([local_idx]*len(mol))

            mol_Xs = np.concatenate(mol_Xs, axis=0)
            mol_yts = None

            if self.all_ys:
                mol_yts = []
                for p_idx in perm[s_m_idx:e_m_idx]:
                    mol_yts.append(self.all_ys[p_idx])

            # print("yielding", mol_Xs[0])

            yield mol_Xs, mol_ids, mol_yts
