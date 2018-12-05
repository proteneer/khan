import tensorflow as tf
import numpy as np

class AtomNN():

    def __init__(self, features, layer_sizes, precision, activation_fn, atom_type="", prefix=""):
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

        precision: tf.dtype
            Should be either tf.float32 or tf.float64        

        activation_fn: tf function
            Examples are tf.nn.relu, or other functions in activations.py

        atom_type: str
            The type of atom we're consdering

        prefix: str
            A prefix we append to the beginning of the variable names

        """
        assert (precision is tf.float32) or (precision is tf.float64)

        assert layer_sizes[-1] == 1

        self.features = features
        self.Ws = []
        self.bs = []
        self.As = [features] # neurons

        self.atom_type = atom_type

        for idx in range(1, len(layer_sizes)):

            x, y = layer_sizes[idx-1], layer_sizes[idx] # input/output
            name = atom_type+"_"+str(x)+"x"+str(y)+"_l"+str(idx)

            if idx != len(layer_sizes) - 1:
                activation = activation_fn
            else:
                activation = None

            A = tf.layers.dense(
                inputs=self.As[-1],
                units=y,
                use_bias=True,
                activation=activation,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=(2.0/(x+y))**0.5),
                kernel_regularizer=tf.keras.regularizers.l2(),
                name=name)

            self.As.append(A)

    def get_parameters(self):
        return self.Ws + self.bs

    def atom_energies(self):
        """
        Retriever the tf.Tensor corresponding to the energies of each atom.

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
        atom_type_features,
        gather_idxs,
        layer_sizes,
        precision,
        activation_fn,
        prefix):
        """
        Construct a molecule neural network that can predict energies of batches of molecules.

        Parameters
        ----------
        type_map: list of str
            Maximum number of atom types. Eg: ["H", "C", "N", "O"]

        atom_type_features: list of tf.Tensors
            Features for each atom_type.

        gather_idxs: tf.Tensor
            Tensor of shape (num_atoms,) of dtype int32 such that a[gather_idx[i]] = original_pos

        mol_idxs: tf.Tensor
            Tensor of shape (num_atoms,) of dtype int32 such that mol[i] is the molecule index in the
            gathered atom list.

        layer_sizes: list of ints
            See documentation of AtomNN for details.

        activation_fn: tf function
            Examples are tf.nn.relu, or other functions in activations.py

        prefix: str
            A prefix we append to the beginning of the variable names


        """

        max_atom_types = len(type_map)
        atom_type_nrgs = [] # len of type-map, one batch of energies for each atom_type
        self.feats = atom_type_features
        self.anns = []

        for type_idx, atom_type in enumerate(type_map):
            ann = AtomNN(atom_type_features[type_idx], layer_sizes, precision, activation_fn,
                atom_type=atom_type, prefix=prefix)
            self.anns.append(ann)

            atom_type_nrgs.append(ann.atom_energies())


        self.atom_outputs = tf.gather(tf.concat(atom_type_nrgs, axis=0), gather_idxs)
        self.atom_outputs = tf.reshape(self.atom_outputs, (-1, )) # (batch_size,)
        # self.mol_nrgs = tf.reshape(tf.segment_sum(self.atom_nrgs, mol_idxs), (-1,))

    def get_parameters(self):
        all_params = []
        for a in self.anns:
            all_params.extend(a.get_parameters())
        return all_params

    def predict_op(self):
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
        return self.mol_nrgs
