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
        self.uncertainty_Ws = []
        self.uncertainty_bs = []
        self.uncertainty_As = []

        self.atom_type = atom_type

        interception_count = 2

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
                    #tf.truncated_normal_initializer(mean=0, stddev=1.0/x**0.5), #(1.0/x**0.5 if idx<len(layer_sizes)-1 else 0.01/x**0.5) ),
                    tf.random_normal_initializer(mean=0, stddev=(2.0/(x+y))**0.5), # maybe a better spread of params without truncation - bad to have any the same, because symmetry could make networks hard to train. max_norm=1.0 should keep the starting error vaguely under control. 
                    trainable=True
                )
                b = tf.get_variable(
                    prefix+"b"+name,
                    (y),
                    precision,
                    tf.zeros_initializer,
                    trainable=True
                )

                # interception layers
                if idx >= len(layer_sizes) - interception_count:
                    u_W = tf.get_variable(
                        prefix+"u_W"+name,
                        (x, y),
                        precision,
                        tf.random_normal_initializer(mean=0, stddev=(2.0/(x+y))**0.5),
                        trainable=True
                    )
                    u_b = tf.get_variable(
                        prefix+"u_b"+name,
                        (y),
                        precision,
                        tf.zeros_initializer,
                        trainable=True
                    )

            A = tf.matmul(self.As[-1], W) + b
            if idx != len(layer_sizes) - 1:
                A = activation_fn(A)



            if idx == len(layer_sizes) - interception_count:
                u_A = tf.matmul(self.As[-1], u_W) + u_b
                if idx != len(layer_sizes) - 1:
                    u_A = activation_fn(u_A)
                else:
                    u_A = tf.nn.softplus(u_A)


            elif idx > len(layer_sizes) - interception_count:
                u_A = tf.matmul(self.uncertainty_As[-1], u_W) + u_b
                if idx != len(layer_sizes) - 1:
                    u_A = activation_fn(u_A)
                else:
                    u_A = tf.nn.softplus(u_A)


                print(A.shape, u_A.shape)

            self.Ws.append(W)
            self.bs.append(b)
            self.As.append(A)

            if idx >= len(layer_sizes) - interception_count:
                self.uncertainty_Ws.append(u_W)
                self.uncertainty_bs.append(u_b)
                self.uncertainty_As.append(u_A)

    def get_parameters(self):
        return self.Ws + self.bs + self.uncertainty_Ws + self.uncertainty_bs

    def atom_energies(self):
        """
        Retriever the tf.Tensor corresponding to the energies of each atom.

        Returns
        -------
        tf.Tensor
            A tensor of shape (num_atoms,) of computed energies

        """
        return self.As[-1]

    def atom_uncertainties(self):
        """
        Retriever the tf.Tensor corresponding to the energies of each atom.

        Returns
        -------
        tf.Tensor
            A tensor of shape (num_atoms,) of computed energies

        """
        return self.uncertainty_As[-1]


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
        atom_type_uncertainties = []
        self.feats = atom_type_features
        self.anns = []

        for type_idx, atom_type in enumerate(type_map):
            ann = AtomNN(atom_type_features[type_idx], layer_sizes, precision, activation_fn,
                atom_type=atom_type, prefix=prefix)
            self.anns.append(ann)

            atom_type_nrgs.append(ann.atom_energies())
            atom_type_uncertainties.append(ann.atom_uncertainties())


        self.atom_outputs = tf.gather(tf.concat(atom_type_nrgs, axis=0), gather_idxs)
        self.atom_outputs = tf.reshape(self.atom_outputs, (-1, )) # (batch_size,)

        self.atom_uncertainties = tf.gather(tf.concat(atom_type_uncertainties, axis=0), gather_idxs)

        self.atom_uncertainties = tf.reshape(self.atom_uncertainties, (-1, )) # (batch_size,)
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
