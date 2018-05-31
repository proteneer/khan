import math

import numpy as np
import tensorflow as tf

class RawDataset():

    def __init__(self, all_Xs, all_ys=None):
        """
        Construct a raw, unfeaturized dataset representing a collection of molecules and
        optionaly their corresponding y values.

        Params:
        -------

        all_Xs: list of np.array
            List of molecular atom types and coordinates, where each element in the list is a rank-2
            Nx4 numpy array, such that the second rank is (type, x, y, z).
        
        all_ys: np.array (optional)
            rank-1 numpy array representing the value of each molecule in the all_Xs array. This is
            not needed when doing inference.

        Example:
        -------

        Xs = [
            np.array([
                [0, 1.0, 2.0, 3.0], # H
                [2, 2.0, 1.0, 4.0], # N
                [0, 0.5, 1.2, 2.3], # H
                [1, 0.3, 1.7, 3.2], # C
                [2, 0.6, 1.2, 1.1], # N
                [0, 14.0, 23.0, 15.0], # H
                [0, 2.0, 0.5, 0.3], # H
                [0, 2.3, 0.2, 0.4]  # H
            ]), # mol0
            np.array([
                [0, 2.3, 0.2, 0.4], # H
                [1, 0.3, 1.7, 3.2], # C
                [2, 0.6, 1.2, 1.1], # N
            ])  # mol1
        ]

        ys = np.array([1.5, 3.3])

        dataset = RawDataset(Xs, ys)

        for xs, m_ids, ys in dataset.iterate(2, batch_size=4, shuffle=True):
            print(xs) => mol1 then mol 0
            print(m_ids) => [1,1,1,0,0,0,0,0]
            print(ys) => [3.3, 1.5]

        .. note:: It is very straightforward to simply construct classes isomorphic to the RawDataset
            API, so long as the num_mols(), num_batches(), an iterate() methods are implemented.

        """
        self.all_ys = all_ys
        self.all_Xs = all_Xs

    def num_mols(self):
        """
        Get the total number of molecules in the dataset.

        Returns
        -------
        int
            the number of molecules in the dataset
        
        """
        return len(self.all_Xs)

    def num_batches(self, batch_size):
        """
        Get the total number of batches in the dataset.

        Parameters
        ----------
        batch_size: int
            Size of the individual batch

        Returns
        -------
        int
            the number of batches in the dataset
        
        """
        return math.ceil(self.num_mols() / batch_size)

    def iterate(self, batch_size, shuffle):
        """
        Generate batches of data of a fixed size.

        Parameters
        ----------
        batch_size: int
            Size of the batch during each iteration.

        shuffle: bool
            If True then we do complete mixing of the dataset, else a stable consecutive
            ordering is assumed.

        Yields
        ------
        3-tuple
            Returns a batch of (compressed coordinates, mol_ids, labels). If the dataset
            does not contain y-values then this returns (compressed coordinates, mol_ids, None)

        .. note::  This does not pad the last batch, so the size of the last batch is just
        the remainder num_mols % batch_size.

        """

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
                mol = self.all_Xs[p_idx]
                mol_Xs.append(mol)
                mol_ids.extend([local_idx]*len(mol))

            mol_Xs = np.concatenate(mol_Xs, axis=0)
            mol_yts = None

            if self.all_ys is not None:
                mol_yts = []
                for p_idx in perm[s_m_idx:e_m_idx]:
                    mol_yts.append(self.all_ys[p_idx])
            yield mol_Xs, mol_ids, mol_yts
