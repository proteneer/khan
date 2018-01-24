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
            x, y = layer_sizes[idx-1], layer_sizes[idx] # input/output
            name = "_"+atom_type+"_"+str(x)+"x"+str(y)+"_l"+str(idx)

            W = tf.get_variable("W"+name, (x, y), np.float32, tf.random_normal_initializer(mean=0, stddev=1.0/x))
            b = tf.get_variable("b"+name, (y), np.float32, tf.zeros_initializer)


            A = tf.matmul(self.As[-1], W) + b
            if idx != len(layer_sizes) - 1:
                # print("EXP")
                A = tf.exp(-A * A)
            # else:
                # print("NORM")

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

def mnn_staging():

    f0_enq = tf.placeholder(dtype=tf.float32)
    f1_enq = tf.placeholder(dtype=tf.float32)
    f2_enq = tf.placeholder(dtype=tf.float32)
    f3_enq = tf.placeholder(dtype=tf.float32)
    gi_enq = tf.placeholder(dtype=tf.int32)
    mi_enq = tf.placeholder(dtype=tf.int32)
    yt_enq = tf.placeholder(dtype=tf.float32)

    staging = tf.contrib.staging.StagingArea(
        capacity=10, dtypes=[
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.int32,
            tf.int32,
            tf.float32])

    put_op = staging.put([f0_enq, f1_enq, f2_enq, f3_enq, gi_enq, mi_enq, yt_enq])
    get_op = staging.get()

    # feat_size = 768

    f0_deq, f1_deq, f2_deq, f3_deq, gi_deq, mi_deq, yt_deq = get_op[0], get_op[1], get_op[2], get_op[3], get_op[4], get_op[5], get_op[6]

    return [
        (f0_enq, f1_enq, f2_enq, f3_enq, gi_enq, mi_enq, yt_enq),
        (f0_deq, f1_deq, f2_deq, f3_deq, gi_deq, mi_deq, yt_deq),
        put_op
    ]


class MoleculeNN():

    def __init__(
        self,
        type_map,
        atom_type_features,
        gather_idxs,
        mol_idxs,
        layer_sizes):
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
            ann = AtomNN(atom_type_features[type_idx], layer_sizes, atom_type)
            self.anns.append(ann)
            atom_type_nrgs.append(ann.atom_energies())

        atom_nrgs = tf.concat(atom_type_nrgs, axis=0)
        self.atom_nrgs = tf.gather(atom_nrgs, gather_idxs)
        self.mol_nrgs = tf.reshape(tf.segment_sum(self.atom_nrgs, mol_idxs), (-1,))

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
