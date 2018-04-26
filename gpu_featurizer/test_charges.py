import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import sparse_ops

ani_mod = tf.load_op_library('ani.so');

@ops.RegisterGradient("AniCharge")
def _ani_charge_grad(op, grads):
    """The gradients for `ani_charge`.

    Args:
    op: The `ani_charge` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `ani_charge` op.

    Returns:
    Gradients with respect to the input of `ani_charge`.
    """

    x,y,z,qs,mo,macs = op.inputs
    # dLdy = grads
    dydx = ani_mod.ani_charge_grad(x,y,z,qs,mo,macs,grads)
    result = [
        None,
        None,
        None,
        dydx,
        None,
        None,
    ]

    return result


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

    scatter_idxs, gather_idxs, atom_counts = ani_mod.ani_sort(atom_types)

    mol_atom_counts = tf.segment_sum(tf.ones_like(mol_idxs), mol_idxs)
    mol_offsets = tf.cumsum(mol_atom_counts, exclusive=True)

    qs_ph = tf.placeholder(dtype=np.float32)

    qs_np = np.array([
        1.0,
        -1.1,
        1.2,
        2.3,
        -4.3,
        3.1,
        0.4,
        -0.9,
        -1.0,
        2.2,
        3.5
    ])

    ys = ani_mod.ani_charge(x,y,z,qs_ph,mol_offsets, mol_atom_counts)
    print(sess.run(ys, feed_dict={qs_ph: qs_np}))



    # grads = ani_mod.ani_charge_grad(
        # x,y,z,qs,mol_offsets, mol_atom_counts)
    # print(sess.run(grads))

    grad = tf.gradients(ys, qs_ph)
    print(sess.run(grad, feed_dict={qs_ph: qs_np}))