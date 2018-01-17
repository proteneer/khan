import tensorflow as tf
import numpy as np


class Symmetrizer():

    def __init__(self,
        R_Rs=(
            5.0000000e-01,
            7.5625000e-01,
            1.0125000e+00,
            1.2687500e+00,
            1.5250000e+00,
            1.7812500e+00,
            2.0375000e+00,
            2.2937500e+00,
            2.5500000e+00,
            2.8062500e+00,
            3.0625000e+00,
            3.3187500e+00,
            3.5750000e+00,
            3.8312500e+00,
            4.0875000e+00,
            4.3437500e+00), # rename to R_Rs, A_Rs, etc.
        R_Rc=4.6,
        R_eta=16,
        A_Rs=(
            5.0000000e-01,
            1.1500000e+00,
            1.8000000e+00,
            2.4500000e+00),
        A_Rc=3.1,
        A_eta=6.0,
        A_thetas=(
            0.0000000e+00,
            7.8539816e-01,
            1.5707963e+00,
            2.3561945e+00,
            3.1415927e+00,
            3.9269908e+00,
            4.7123890e+00,
            5.4977871e+00),
        A_zeta=8.0):
        """
        Symmetrizer implements the ANI symmetry functions as described in doi:10.1039/C6SC05720A
        In particular, the radial_symmetry function implements eq (3) and the angular_symmetry
        function implements eq (4) of the paper. Note: Unless mentioned explicitly, all floating
        point numbers should be float32s.

        Parameters
        ----------
        R_Rs: array-like of floats
            Elements corresponding to radial spikes

        R_Rc: float
            Cutoff for the radial part of featurization

        R_eta: float
            Eta parameter for the radial part of featurization

        A_Rs: array-like of floats
            Elements corresponding to angular spikes

        A_Rc: float
            Cutoff for the angular part of featurization

        A_eta: float
            Eta parameter for the angular part of featurization

        A_theta: array-like of floats
            Parameters determining the angular separation

        A_zeta: float
            Angular exponent (typically used to normalize to the radial parts)


        """

        self.max_atom_types = 4 # CNOH

        self.R_Rs = np.array(R_Rs, dtype=np.float32)
        self.R_Rc = np.float32(R_Rc)
        self.R_eta = np.float32(R_eta)

        self.A_Rs = np.array(A_Rs, dtype=np.float32)
        self.A_Rc = np.float32(A_Rc)
        self.A_eta = np.float32(A_eta)
        self.A_thetas = np.array(A_thetas, dtype=np.float32)
        self.A_zeta = np.float32(A_zeta)

        self.gdb_sess = tf.Session()

    def featurize_one(self, atom_matrix):
        """
        Featurize one molecule using both the radial and angular symmetry functions.

        Parameters
        ----------
        atom_matrix: tf.Tensor
            An atom matrix of shape (None, 4), where rank 0 determines the number of atoms
            and rank 1 consists of (t, x, y, z) such that t is a compacted atomic number.

        Returns
        -------
        tf.Tensor
            Returns a fully featurized rank 2 tensor of shape (num_atoms, and tot_features).
            tot_features is the concatenated result of radial and angular features.

        """
        return tf.concat([self.radial_symmetry(atom_matrix), self.angular_symmetry(atom_matrix)], axis=1)
    
    def featurize_batch(self, batched_atom_matrix, offsets):
        """
        Featurize a batch of atom matrices.

        Parameters
        ----------
        batched_atom_matrix: tf.Tensor
            Atom matrices of shape (None, 4), representing compacted molecules.
    
        offsets: tf.Tensor
            Tensor of shape (None, 2), where rank 1 indicates the start and end such that
            batch_atom_matrix[start:end, :] is a molecule.

        Returns
        -------
        tf.Tensor
            Returns a completely featurized representation of the input molecules.

        """
        num_mols = tf.shape(offsets)[0]

        ta_result = tf.TensorArray(
            dtype=tf.float32,
            size=tf.shape(batched_atom_matrix)[0],
            name="ta_results",
            element_shape=(self.feature_size()),
        )

        def compute(idx, accum):
            start_idx = offsets[idx][0]
            end_idx = offsets[idx][1]
            mol = batched_atom_matrix[start_idx:end_idx, :]
            accum = accum.scatter(tf.range(start=start_idx, limit=end_idx), self.featurize_one(mol))
            return (idx+1, accum)

        _, ta_results = tf.while_loop(
            lambda i, _: i < num_mols, # condition
            compute, # body
            (0, ta_result)) # initial_state

        return ta_results.stack()

    def radial_feature_size(self):
        return np.int32(self.max_atom_types * len(self.R_Rs))

    def angular_feature_size(self):
        return np.int32(len(self.A_Rs)*len(self.A_thetas)*self.max_atom_types*(self.max_atom_types+1)/2)

    def feature_size(self):
        """
        Return the length of the featurized representation. This is equal to the sum of the radial and
        angular lengths.

        Returns
        -------
        int32
            Length of the resulting features

        """
        return self.radial_feature_size() + self.angular_feature_size()

    def radial_symmetry(self, atom_matrix):
        """
        Generate radial basis functions given an atom_matrix consisting of the atom types
        and coordinates.

        Parameters
        ----------
        atom_matrix: tf.Tensor
            An atom matrix of shape (None, 4), where rank 0 determines the number of atoms
            and rank 1 consists of (t, x, y, z) such that t is a compacted atomic number.

        Returns
        -------
        tf.Tensor
            Returns a tensor of shape (num_atoms, max_atom_types * len(self.Rs))

        """
        num_atoms = tf.shape(atom_matrix)[0]
        atom_types = tf.cast(atom_matrix[:, 0], dtype=tf.int32)
        atom_coords = atom_matrix[:, 1:]
        type_groups = tf.dynamic_partition(atom_coords, atom_types, self.max_atom_types, name="dp_radial")
        atom_coords = tf.expand_dims(atom_coords, axis=1) #  we reshape atom_coords to (num_atoms, 1, 3)
        # loop through each of the atom types, eg. find all the atoms of type H, C, N, O, etc.
        radial_features = []
        for grp_pos in type_groups:
            # grp_stack shape: (num_atoms, ?, 3)
            # grp_pos shape: (grp_atoms, 3) 
            grp_pos = tf.expand_dims(grp_pos, axis=0)            
            R_ij = tf.norm(atom_coords - grp_pos, axis=2)

            # R_ij shape: (num_atoms, ?), where ? is the number of matching atoms

            f_C_true = 0.5*tf.cos(tf.div(np.pi * R_ij, self.R_Rc)) + 0.5
            f_C_flags = tf.nn.relu(tf.sign(self.R_Rc - R_ij)) # 1 if within cutoff, 0 otherwise
            f_C = f_C_true * f_C_flags
            f_C = tf.expand_dims(f_C, 2)

            delta_Rs = tf.expand_dims(R_ij, axis=-1) -  tf.reshape(self.R_Rs, (1, 1, -1))
            # this is equivalent to the following:
            # delta_Rs = tf.stack([R_ij] * len(self.R_Rs), axis=2) - self.R_Rs

            summand = tf.exp(-self.R_eta*tf.pow(delta_Rs, 2)) * f_C

            # summand is of shape (num_atoms, group_count, len(self.Rs))
            # (ytz): a trick to determine if R_ij is such that i == j, we make the if and only if
            # assumption that R_ij < 1e-6. 
            R_ij_flags = tf.abs(R_ij) # deals with numerical inprecisions in cases of -1e6
            R_ij_flags = tf.nn.relu(tf.sign(R_ij_flags - 1e-6))
            R_ij_flags = tf.expand_dims(R_ij_flags, 2)

            radial_features.append(tf.reduce_sum(summand * R_ij_flags, axis=1))

        radial_features = tf.concat(radial_features, axis=1)
        radial_features = tf.reshape(radial_features, (num_atoms, self.radial_feature_size())) # ravel

        return radial_features

    def angular_symmetry(self, atom_matrix):
        """
        Generate radial basis functions given an atom_matrix consisting of the atom types
        and coordinates.

        Parameters
        ----------
        atom_matrix: tf.Tensor
            An atom matrix of shape (None, 4), where rank 0 determines the number of atoms
            and rank 1 consists of (t, x, y, z) such that t is a compacted atomic number.

        Returns
        -------
        tf.Tensor
            Featurized representation of shape (num_atoms, len(sym.A_Rs)*len(sym.A_thetas)*sym.max_atom_types*(sym.max_atom_types+1)/2)

        """
        num_atoms = tf.shape(atom_matrix)[0]
        atom_idxs = tf.range(tf.shape(atom_matrix)[0])
        atom_types = tf.cast(atom_matrix[:, 0], dtype=tf.int32)
        atom_coords = atom_matrix[:, 1:] # atom_coords shape: (num_atoms, 3)
        type_groups_idxs = tf.dynamic_partition(atom_idxs, atom_types, self.max_atom_types, name="dp_angular")
        lookup = np.array([[[0, 3]],[[2,3]],[[5,3]]])

        angular_features = []
        for type_a in range(self.max_atom_types):
            j_idxs = type_groups_idxs[type_a]

            for type_b in range(type_a, self.max_atom_types):
                k_idxs = type_groups_idxs[type_b]

                tile_a = tf.tile(tf.expand_dims(j_idxs, 1), [1, tf.shape(k_idxs)[0]], name="tile_outer1")
                tile_a = tf.expand_dims(tile_a, 2) 
                tile_b = tf.tile(tf.expand_dims(k_idxs, 0), [tf.shape(j_idxs)[0], 1], name="tile_outer2")
                tile_b = tf.expand_dims(tile_b, 2) 
                cartesian_product = tf.concat([tile_a, tile_b], axis=2) # int64s?
                
                group_coords = tf.nn.embedding_lookup(atom_coords, cartesian_product) # shape: (len(type_a), len(type_b), 2, 3)
                delta_jk = group_coords[:, :, 0, :] - group_coords[:, :, 1, :]
                R_jk = tf.norm(delta_jk, axis=-1)

                dist_vec = tf.reshape(atom_coords, (-1, 1, 1, 1, 3)) # shape (6, 3, 3, 2, 3), vector difference

                deltas = group_coords - dist_vec # shape: (num_atoms, len(type_a), len(type_b), 2, 3)
                delta_ij = deltas[:, :, :, 0, :]
                delta_ik = deltas[:, :, :, 1, :]

                # LHS computation
                denom = tf.multiply(tf.norm(delta_ij, axis=-1), (tf.norm(delta_ik, axis=-1))) #
                dot = tf.reduce_sum(tf.multiply(delta_ij, delta_ik), axis=-1)
                theta_ijk = tf.acos(dot / denom)   # if i=j || j=k then NaN

                lhs = tf.pow(1 + tf.cos(tf.expand_dims(theta_ijk, -1) - tf.reshape(self.A_thetas, (1, 1, 1, -1))), self.A_zeta)
                lhs = tf.where(tf.is_nan(lhs), tf.zeros_like(lhs), lhs) # clean up nans numerically, the real zeroing happens later
                lhs = tf.where(tf.is_inf(lhs), tf.zeros_like(lhs), lhs) # clean up infs numerically, the real zeroing happens later
                
                # RHS computation
                R_ij_ik = tf.norm(deltas, axis=-1) # shape (6, 3, 3, 2), norm distance
                f_C_true = 0.5*tf.cos(tf.div(np.pi * R_ij_ik, self.A_Rc)) + 0.5 # TODO: refactor with radial code?
                f_C_flags = tf.nn.relu(tf.sign(self.A_Rc - R_ij_ik)) # 1 if within cutoff, 0 otherwise
                f_C_R_ij_ik  = f_C_true * f_C_flags

                # note: element wise multiply
                fCRi_fCRj = tf.multiply(f_C_R_ij_ik[:, :, :, 0], f_C_R_ij_ik[:, :, :, 1])
                R_ij = R_ij_ik[:, :, :, 0]
                R_ik = R_ij_ik[:, :, :, 1]

                inner = tf.expand_dims((R_ij + R_ik) / 2.0, -1) - tf.reshape(self.A_Rs, (1, 1, 1, -1))
                rhs = tf.exp(-self.A_eta*tf.pow(inner, 2)) * tf.expand_dims(fCRi_fCRj, -1)

                # lhs shape: [num_atoms, len(type_a), len(type_b), len(A_thetas)]
                # rhs shape: [num_atoms, len(type_a), len(type_b), len(A_Rs)]
                lhs = tf.expand_dims(lhs, axis=3)
                rhs = tf.expand_dims(rhs, axis=4)
                summand = tf.multiply(lhs, rhs) # (num_atoms, len(type_a), len(type_b), len(A_Rs), len(A_thetas))

                # zero-out/fix summand elements where i == j || j == k || i == k
                # we store a triplet of shape
                # (num_atoms, len(type_a), len(type_b), 3) where 3 is the distance of ij, ik, and jk respectively
                # R_ij shape: (num_atoms, len(type_a), len(type_b))
                # R_ik shape: (num_atoms, len(type_a), len(type_b))
                R_jk = tf.tile(tf.expand_dims(R_jk, axis=0), [num_atoms, 1, 1], name="tile_inner")
                R_ijk = tf.stack([R_ij, R_ik, R_jk], axis=-1)

                # R_jk shape: (len(type_a), len(type_b))
                # We want to form R_ijk of shape (num_atoms, len(type_a), len(type_b), 3)
                min_dists = tf.reduce_min(R_ijk, axis=-1)
                keep_flags = tf.nn.relu(tf.sign(tf.abs(min_dists) - 1e-7))
                keep_flags = tf.expand_dims(keep_flags, -1)
                keep_flags = tf.expand_dims(keep_flags, -1)

                summand = tf.multiply(summand, keep_flags)
                result = tf.multiply(tf.pow(np.float32(2.0), 1-self.A_zeta), tf.reduce_sum(summand, [1,2])) 
                result = tf.reshape(result, (num_atoms, len(self.A_thetas)*len(self.A_Rs)))

                angular_features.append(result)

        angular_features = tf.concat(angular_features, axis=1)
        angular_features = tf.reshape(angular_features, (num_atoms, self.angular_feature_size())) # ravel

        return angular_features
