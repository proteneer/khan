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

        atom_type: str
            The type of atom we're 

        """
        assert (precision is tf.float32) or (precision is tf.float64)

        assert layer_sizes[-1] == 1

        self.features = features
        self.Ws = []
        self.bs = []
        self.As = [features] # neurons
        self.atom_type = atom_type

        # self.alpha = tf.get_variable(prefix+'_'+atom_type+'_alpha', tuple(), tf.float32, tf.constant_initializer(0.1), trainable=False)

        for idx in range(1, len(layer_sizes)):
            x, y = layer_sizes[idx-1], layer_sizes[idx] # input/output
            # print('Layer', idx, 'input/output size', x, y)
            name = "_"+atom_type+"_"+str(x)+"x"+str(y)+"_l"+str(idx)

            with tf.device('/cpu:0'):
                W = tf.get_variable(
                    prefix+"W"+name,
                    (x, y),
                    precision,
                    tf.random_normal_initializer(mean=0, stddev=(2.0/(x+y))**0.5),
                    trainable=True
                )
                b = tf.get_variable(
                    prefix+"b"+name,
                    (y),
                    precision,
                    tf.zeros_initializer,
                    trainable=True
                )

            A = tf.matmul(self.As[-1], W) + b
            if idx != len(layer_sizes) - 1:
                A = activation_fn(A)

            self.Ws.append(W)
            self.bs.append(b)
            self.As.append(A)

    def get_parameters(self):
        return self.Ws + self.bs

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

def mnn_staging():

    x_enq = tf.placeholder(dtype=tf.float32)
    y_enq = tf.placeholder(dtype=tf.float32)
    z_enq = tf.placeholder(dtype=tf.float32)
    a_enq = tf.placeholder(dtype=tf.int32)
    m_enq = tf.placeholder(dtype=tf.int32)
    si_enq = tf.placeholder(dtype=tf.int32)
    gi_enq = tf.placeholder(dtype=tf.int32)
    ac_enq = tf.placeholder(dtype=tf.int32)
    y_trues = tf.placeholder(dtype=tf.float32)

    staging = tf.contrib.staging.StagingArea(
        capacity=10, dtypes=[
            tf.float32,  # Xs
            tf.float32,  # Ys
            tf.float32,  # Zs
            tf.int32,    # As
            tf.int32,    # mol ids
            tf.int32,    # scatter idxs
            tf.int32,    # gather  idxs
            tf.int32,    # atom counts
            tf.float32   # Y TRUEss
        ])

    put_op = staging.put([x_enq, y_enq, z_enq, a_enq, m_enq, si_enq, gi_enq, ac_enq, y_trues])
    get_op = staging.get()

    return [
        (x_enq,     y_enq,     z_enq,     a_enq,     m_enq,     si_enq,    gi_enq,    ac_enq,    y_trues),
        get_op,
        # (get_op[0], get_op[1], get_op[2], get_op[3], get_op[4], get_op[5], get_op[6], get_op[7], get_op[8]),
        put_op
    ]


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
