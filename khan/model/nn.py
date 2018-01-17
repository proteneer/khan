import tensorflow as tf
import numpy as np

class AtomNN():

    def __init__(self, features, layer_sizes, atom_type=""):
        """
        Construct a Neural Network used to compute energies of atoms.

        Parameters
        ----------
        features: tf.Tensor
            A batch of atom features.

        layer_sizes: list of ints
            A collection of layers denoting neural network layer architecture. The
            0th layer should be the size of the features, while the last layer must
            be exactly of size 1.

        activation_fns: list of functions
            A list of activation

        atom_type: str
            The type of atom we're 

        """

        assert layer_sizes[-1] == 1

        self.features = features
        self.Ws = []
        self.bs = []
        self.As = [features] # neurons
        self.atom_type = atom_type

        for idx in range(1, len(layer_sizes)):
            # initial_W = np.zeros((layer_sizes[idx-1], layer_sizes[idx]), dtype=np.float32)
            # initial_b = np.zeros((layer_sizes[idx],), dtype=np.float32)

            x, y = layer_sizes[idx-1], layer_sizes[idx]
            name = "_"+atom_type+"_"+str(x)+"x"+str(y)+"_l"+str(idx)

            initial_W = tf.get_variable("W"+name, (x, y), np.float32, tf.contrib.layers.xavier_initializer())
            initial_b = tf.get_variable("b"+name, (y), np.float32, tf.contrib.layers.xavier_initializer())
            
            print(initial_W.shape, initial_b.shape)

            W = tf.Variable(initial_W)
            b = tf.Variable(initial_b)
            A = tf.matmul(self.As[-1], W) + b
            if idx != len(layer_sizes) - 1:
                A = tf.exp(-A * A)

            self.Ws.append(W)
            self.bs.append(b)
            self.As.append(A)

    def atom_energies(self):
        """
        Generate one layer in the atomic differentiated dense structure.
        
        Parameters
        ----------
        batch_atom_features: tf.Tensor
            A tensor of shape (num_atoms, feature_size) representing a collection of atomic
            energies to be evaluated.

        Returns
        -------
        tf.Tensor
            A tensor of shape (num_atoms,) of computed energies

        """
        return self.As[-1]


class MoleculeNN():

    def __init__(
        self,
        type_map,
        batched_atom_features,
        offsets,
        layer_sizes):
        """
        Construct a molecule neural network that can predict energies of batches of molecules.

        Parameters
        ----------
        type_map: list of str
            Maximum number of atom types. Eg: ["H", "C", "N", "O"]

        batched_atom_features: tf.Tensor
            Tensor comprised of compacted and featurized molecules.

        offsets:
            Tensor of shape (None, 2), where rank 1 indicates the start and end such that
            batch_atom_matrix[start:end, :] is a featurized molecule.


        layer_sizes: list of ints
            See documentation of AtomNN for details.

        """

        self.max_atom_types = len(type_map)
        self.anns = []

        atom_types = tf.cast(batched_atom_features[:, 0], dtype=tf.int32)
        atom_feats = batched_atom_features[:, 1:]

        group_feats = tf.dynamic_partition(
            atom_feats,
            atom_types,
            num_partitions=self.max_atom_types) # feature shape: (batch_size across all atoms, )

        atom_nrgs = []


        for group_type, feats in enumerate(group_feats):
            # print(layer_sizes)
            ann = AtomNN(feats, layer_sizes, type_map[group_type])
            self.anns.append(ann)
            atom_nrgs.append(ann.atom_energies())

        atom_idxs = tf.dynamic_partition(
            tf.range(tf.shape(atom_types)[0]),
            atom_types,
            num_partitions=self.max_atom_types)

        all_atom_nrgs = tf.dynamic_stitch(atom_idxs, atom_nrgs)

        num_mols = tf.shape(offsets)[0]

        ta_result = tf.TensorArray(
            dtype=tf.float32,
            size=num_mols,
            name="ta_results",
            element_shape=(),
        )

        def compute(idx, accum):
            start_idx = offsets[idx][0]
            end_idx = offsets[idx][1]
            mol_nrg = tf.reduce_sum(all_atom_nrgs[start_idx:end_idx])
            accum = accum.write(idx, mol_nrg)
            return (idx+1, accum)

        _, ta_results = tf.while_loop(
            lambda i, _: i < num_mols, # condition
            compute, # body
            (0, ta_result)) # initial_state

        self.mol_nrgs = ta_results.stack()

    def molecule_energies(self):
        """
        Compute molecular energies for a batch of molecules.

        Parameters
        ----------
        batched_atom_features: tf.Tensor
            A tensor of atom features.

        offsets: tf.Tensor
            Tensor of shape (None, 2), where rank 1 indicates the start and end such that
            batched_atom_features[start:end, :] is a featurized molecule.

        Returns
        -------
        tf.Tensor
            A tensor of shape (num_mols,) of dtype tf.float32 representing the predicted
            energy of the molecule.

        """
        return tf.reduce_sum(self.mol_nrgs)
