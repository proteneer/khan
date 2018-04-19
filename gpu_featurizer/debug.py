import tensorflow as tf
import numpy as np

ani_mod = tf.load_op_library('ani.so');
sort_lib = tf.load_op_library('ani_sort.so');

if __name__ == "__main__":

    cp = tf.ConfigProto(log_device_placement=True, allow_soft_placement=False, device_count = {'GPU': 1})

    sess = tf.Session(config=cp)

    atom_matrix = np.array([
        [0, 1.0, 2.0, 3.0], # H
        [2, 2.0, 1.0, 4.0], # N
        [0, 0.5, 1.2, 2.3], # H
        [1, 0.3, 1.7, 3.2], # C
        [2, 0.6, 1.2, 1.1], # N
        [0, 14.0, 23.0, 15.0], # H
        [0, 2.0, 0.5, 0.3], # H
        [0, 2.3, 0.2, 0.4], # H

        [0, 2.3, 0.2, 0.4], # H
        [1, 0.3, 1.7, 3.2], # C
        [2, 0.6, 1.2, 1.1]], dtype=np.float32)


    mol_idxs = np.array([0,0,0,0,0,0,0,0,1,1,1], dtype=np.int32)

    atom_types = atom_matrix[:, 0]
    x = atom_matrix[:, 1]
    y = atom_matrix[:, 2]
    z = atom_matrix[:, 3]

    scatter_idxs, gather_idxs, atom_counts = sort_lib.ani_sort(atom_types)

    mol_atom_counts = tf.segment_sum(tf.ones_like(mol_idxs), mol_idxs)
    mol_offsets = tf.cumsum(mol_atom_counts, exclusive=True)

    obtained_si, obtained_gi, obtained_ac = sess.run([scatter_idxs, gather_idxs, atom_counts])

    with tf.device('/device:GPU:0'):
        f0, f1, f2, f3 = ani_mod.featurize(
            x,
            y,
            z,
            atom_types,
            mol_offsets,
            mol_atom_counts,
            scatter_idxs,
            atom_counts
        )

        # commenting out the reshape line allows the code to run correctly on the GPU
        f0, f1, f2, f3 = tf.reshape(f0, (-1, 384)), tf.reshape(f1, (-1, 384)), tf.reshape(f2, (-1, 384)), tf.reshape(f3, (-1, 384))

    obtained_features = sess.run([f0, f1, f2, f3])