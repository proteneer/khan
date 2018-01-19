# end to end ANI-1 model.

import tensorflow as tf
from khan.model.nn import MoleculeNN

class ANI():

    def __init__(self,
        atom_type_map,
        symmetrizer,
        batched_atom_matrix,
        mol_offsets,
        mol_idxs,
        gather_idxs, # atom type scatter idxs
        layer_sizes):
        """
        Model for full end-to-end.
        

        Parameters
        ----------
    

        """

        self.bam = batched_atom_matrix
        self.mol_offsets = mol_offsets

        unsorted_atom_feats = symmetrizer.featurize_batch(self.bam, self.mol_offsets)

        atom_types = tf.cast(self.bam[:, 0], dtype=tf.int32)
        # dynamic partition is a stable partition.
        self.atom_type_feats = tf.dynamic_partition(
            unsorted_atom_feats,
            atom_types,
            num_partitions=len(atom_type_map))
        self.gather_idxs = gather_idxs # overridden during training
        self.mol_idxs = mol_idxs # overriden during training

        self.mnn = MoleculeNN(
            atom_type_map,
            self.atom_type_feats,
            self.gather_idxs,
            self.mol_idxs,
            layer_sizes)

        # self.preds = predictions
        self.tau = 0.5
        self.mol_nrgs = self.mnn.molecule_energies()

    def predict_op(self):
        return self.mol_nrgs