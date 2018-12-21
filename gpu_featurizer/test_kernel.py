import unittest
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import os

ani_path = os.path.join(os.getcwd(), 'ani.so')
ani_mod = tf.load_op_library(ani_path)

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
    global ani_mod
    assert ani_mod is not None
    x,y,z,qs,mo,macs = op.inputs
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

@ops.RegisterGradient("Featurize")
def _feat_grad(op, grad_hs, grad_cs, grad_ns, grad_os):
    x,y,z,a,mo,macs,sis,acs = op.inputs

    # print(dir(op))
    # assert 0
    dx, dy, dz = ani_mod.featurize_grad(
        x,
        y,
        z,
        a,
        mo,
        macs,
        sis,
        acs,
        grad_hs,
        grad_cs,
        grad_ns,
        grad_os,
        n_types=op.get_attr("n_types"),
        R_Rc=op.get_attr("R_Rc"),
        R_eta=op.get_attr("R_eta"),
        A_Rc=op.get_attr("A_Rc"),
        A_eta=op.get_attr("A_eta"),
        A_zeta=op.get_attr("A_zeta"),
        R_Rs=op.get_attr("R_Rs"),
        A_thetas=op.get_attr("A_thetas"),
        A_Rs=op.get_attr("A_Rs"))

    return [
        dx,
        dy,
        dz,
        None,
        None,
        None,
        None,
        None,
    ]

@ops.RegisterGradient("FeaturizeGrad")
def _feat_grad_grad(op, dLdx, dLdy, dLdz):
    x,y,z,a,mo,macs,sis,acs,gh,gc,gn,go = op.inputs
    dh, dc, dn, do = ani_mod.featurize_grad_inverse(
        x,
        y,
        z,
        a,
        mo,
        macs,
        sis,
        acs,
        dLdx,
        dLdy,
        dLdz,
        n_types=op.get_attr("n_types"),
        R_Rc=op.get_attr("R_Rc"),
        R_eta=op.get_attr("R_eta"),
        A_Rc=op.get_attr("A_Rc"),
        A_eta=op.get_attr("A_eta"),
        A_zeta=op.get_attr("A_zeta"),
        R_Rs=op.get_attr("R_Rs"),
        A_thetas=op.get_attr("A_thetas"),
        A_Rs=op.get_attr("A_Rs")
    )

    # is this correct?
    return [
        None, # x 
        None, # y
        None, # z
        None, # a
        None, # mo
        None, # macs
        None, # sis
        None, # acs
        dh,
        dc,
        dn,
        do
    ]



def linearize(i, j, k, l):
    if j < i:
        tmp = i
        i = j
        j = tmp

    N = 4
    K = 8
    L = 4

    basis = (N*(N-1)//2 - (N-i) * (N-i-1)//2 +j)
    
    idx = basis*K*L + k*L + l


    return idx



def fC(r_ij, r_C):

    if r_ij <= r_C:
        return 0.5 * math.cos(math.pi*r_ij/r_C) + 0.5
    else:
        return 0.0


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

class TestFeaturizer(unittest.TestCase):

    def setUp(self):

        cp = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 1})
        self.sess = tf.Session(config=cp)

        # self.sess = tf.Session()

    def tearDown(self):

        self.sess.close()

    def reference_feats(self, mol):

        R_Rs = [
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
            4.3437500e+00
        ]

        R_Rc = 4.6
        R_eta = 16

        A_thetas = [
            0.0000000e+00,
            7.8539816e-01,
            1.5707963e+00,
            2.3561945e+00,
            3.1415927e+00,
            3.9269908e+00,
            4.7123890e+00,
            5.4977871e+00
        ]

        A_Rs = [
            5.0000000e-01,
            1.1500000e+00,
            1.8000000e+00,
            2.4500000e+00,
        ]

        A_Rc = 3.1;
        A_eta = 6.0;
        A_zeta = 8.0;

        all_feats = []

        for i_idx, row_i in enumerate(mol):

            radial_feats = np.zeros(len(R_Rs) * 4, dtype=np.float32)
            angular_feats = np.zeros(len(A_Rs) * len(A_thetas) * (4 * (4+1) // 2), dtype=np.float32)
            # print(angular_feats.shape)

            i_xyz = row_i[1:]
            i_type = int(row_i[0])

            for j_idx, row_j in enumerate(mol):

                if i_idx == j_idx:
                    continue

                j_xyz = row_j[1:]
                j_type = int(row_j[0])
                r_ij = np.linalg.norm(i_xyz - j_xyz)

                for r_idx, r_s in enumerate(R_Rs):
                    # if r_ij < R_Rc and i_idx < j_idx:
                    fc = fC(r_ij, R_Rc) 
                    lhs = math.exp(-R_eta*math.pow(r_ij-r_s, 2))
                    summand = fc * lhs
                    radial_feats[j_type*len(R_Rs)+r_idx] += summand

                for k_idx, row_k in enumerate(mol):

                    if i_idx == k_idx:
                        continue

                    if j_idx >= k_idx:
                        continue

                    k_xyz = row_k[1:]
                    k_type = int(row_k[0])
                    r_ik = np.linalg.norm(i_xyz - k_xyz)
                    v_ij = i_xyz - j_xyz
                    v_ik = i_xyz - k_xyz

                    theta_ijk = angle(v_ij, v_ik)

                    if r_ij < A_Rc and r_ik < A_Rc:

                        for at_idx, at_t in enumerate(A_thetas):
                            for ar_idx, ar_s in enumerate(A_Rs):

                                fC_ij = fC(r_ij, A_Rc)
                                fC_ik = fC(r_ik, A_Rc)


                                summand = math.pow(1+math.cos(theta_ijk - at_t), A_zeta) * math.exp(-A_eta*math.pow((r_ij+r_ik)/2 - ar_s, 2)) * fC_ij * fC_ik
                                summand = pow(2, 1 - A_zeta) * summand

                                # print(j_idx, k_idx, at_idx, ar_idx)
                                # print("i,j,k,t,s", i_idx, j_idx, k_idx, at_idx, ar_idx, linearize(j_type, k_type, at_idx, ar_idx))
                                angular_feats[linearize(j_type, k_type, at_idx, ar_idx)] += summand


            all_feats.append(np.concatenate([radial_feats, angular_feats]))

        return np.array(all_feats)


    def test_featurizer(self):


        precisions = [tf.float32, tf.float64]
        # precisions = [tf.float64]
        # precisions = [tf.float32]
        with self.sess:

            for prec in precisions:

                # 8 + 3 atoms
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
                    [2, 0.6, 1.2, 1.1], # N
                    ], dtype=prec.as_numpy_dtype)


                mol_idxs = np.array([0,0,0,0,0,0,0,0,1,1,1], dtype=np.int32)

                atom_types = atom_matrix[:, 0].astype(np.int32)
                x = atom_matrix[:, 1]
                y = atom_matrix[:, 2]
                z = atom_matrix[:, 3]

                ph_atom_types = tf.placeholder(dtype=np.int32, name="atom_types")
                ph_xs = tf.placeholder(dtype=prec, name="xs")
                ph_ys = tf.placeholder(dtype=prec, name="ys")
                ph_zs = tf.placeholder(dtype=prec, name="zs")
                ph_mol_idxs = tf.placeholder(dtype=np.int32, name="mol_idxs")

                scatter_idxs, gather_idxs, atom_counts = ani_mod.ani_sort(ph_atom_types)

                mol_atom_counts = tf.segment_sum(tf.ones_like(ph_mol_idxs), ph_mol_idxs)
                mol_offsets = tf.cumsum(mol_atom_counts, exclusive=True)

                obtained_si, obtained_gi, obtained_ac = self.sess.run(
                    [scatter_idxs, gather_idxs, atom_counts],
                    feed_dict={
                        ph_mol_idxs: mol_idxs,
                        ph_atom_types: atom_types,
                    })

                np.testing.assert_array_equal(obtained_ac, [6,2,3,0])
                np.testing.assert_array_equal(obtained_si, [0,0,1,0,1,2,3,4,5,1,2])
                np.testing.assert_array_equal(obtained_gi, [0,8,1,6,9,2,3,4,5,7,10])

                f0, f1, f2, f3 = ani_mod.featurize(
                    ph_xs,
                    ph_ys,
                    ph_zs,
                    ph_atom_types,
                    mol_offsets,
                    mol_atom_counts,
                    scatter_idxs,
                    atom_counts,
                    n_types=4,
                    R_Rc=4.6,
                    R_eta=16.0,
                    A_Rc=3.1,
                    A_eta=6.0,
                    A_zeta=8.0,
                    R_Rs=[5.0000000e-01,7.5625000e-01,1.0125000e+00,1.2687500e+00,1.5250000e+00,1.7812500e+00,2.0375000e+00,2.2937500e+00,2.5500000e+00,2.8062500e+00,3.0625000e+00,3.3187500e+00,3.5750000e+00,3.8312500e+00,4.0875000e+00,4.3437500e+00],
                    A_thetas=[0.0000000e+00,7.8539816e-01,1.5707963e+00,2.3561945e+00,3.1415927e+00,3.9269908e+00,4.7123890e+00,5.4977871e+00],
                    A_Rs=[5.0000000e-01,1.1500000e+00,1.8000000e+00,2.4500000e+00],
                )

                FEATURE_SIZE = 384

                f0, f1, f2, f3 = tf.reshape(f0, (-1, FEATURE_SIZE)), tf.reshape(f1, (-1, FEATURE_SIZE)), tf.reshape(f2, (-1, FEATURE_SIZE)), tf.reshape(f3, (-1, FEATURE_SIZE))
            
                scattered_features = tf.concat([f0, f1, f2, f3], axis=0)
                features = tf.gather(scattered_features, gather_idxs)
                    
                obtained_features = self.sess.run(features, feed_dict={
                    ph_xs: x,
                    ph_ys: y,
                    ph_zs: z,
                    ph_mol_idxs: mol_idxs,
                    ph_atom_types: atom_types
                })

                expected_features_mol1 = self.reference_feats(atom_matrix[:8, :])
                np.testing.assert_almost_equal(obtained_features[:8, :64], expected_features_mol1[:, :64], decimal=6)
                expected_features_mol2 = self.reference_feats(atom_matrix[8:, :])
                np.testing.assert_almost_equal(obtained_features[8:, :64], expected_features_mol2[:, :64], decimal=6)

                # angular components
                np.testing.assert_almost_equal(obtained_features[:8, 64:], expected_features_mol1[:, 64:], decimal=6)
                np.testing.assert_almost_equal(obtained_features[8:, 64:], expected_features_mol2[:, 64:], decimal=6)





                if prec == tf.float32:
                    delta = 1e-3
                    tol = 2e-3
                elif prec == tf.float64:
                    delta = 1e-3
                    tol = 1e-11
                else:
                    raise Exception("Unknown precision")

                # some parameter set
                ph_t = tf.placeholder(dtype=prec)

                E = tf.multiply(features, ph_t)

                # random set of parameters
                # np.random.seed(seed=1)
                t = np.random.rand(1, 384)
                forces_x = tf.gradients(E, [ph_xs])[0]
                force_loss = tf.reduce_sum(forces_x)
                debug_grad = tf.gradients(force_loss, ph_t)

                res = self.sess.run(debug_grad, feed_dict={
                    ph_xs: x,
                    ph_ys: y,
                    ph_zs: z,
                    ph_mol_idxs: mol_idxs,
                    ph_atom_types: atom_types,
                    ph_t: t
                })

                error = tf.test.compute_gradient_error(
                    ph_t,
                    (1, 384),
                    force_loss,
                    (1,),
                    x_init_value=t,
                    delta=delta,
                    extra_feed_dict={
                        ph_xs: x,
                        ph_ys: y,
                        ph_zs: z,
                        ph_mol_idxs: mol_idxs,
                        ph_atom_types: atom_types
                    }
                )

                print(prec, "ddx", error, tol)
                assert error < tol


                if prec == tf.float32:
                    delta = 1e-3
                    tol = 2e-3
                elif prec == tf.float64:
                    delta = 1e-3
                    tol = 1e-11
                else:
                    raise Exception("Unknown precision")

                # test charges (CURRENTLY BROKEN FOR 64bit)
                # ph_qs = tf.placeholder(dtype=prec);

                # charge_energy = ani_mod.ani_charge(
                #     ph_xs,
                #     ph_ys,
                #     ph_zs,
                #     ph_qs,
                #     mol_offsets,
                #     mol_atom_counts
                # )


                # # with self.sess:
                # qs = np.array([
                #     0.3,
                #     4.5,
                #     2.2,
                #     3.4,
                #     -5.6,
                #     3.4,
                #     1.1,
                #     3.2,

                #     1.0,
                #     0.3,
                #     2.4], dtype=prec.as_numpy_dtype)


                # # test dL/dq
                # error = tf.test.compute_gradient_error(
                #     ph_qs,
                #     (11,),
                #     charge_energy,
                #     (2,),
                #     x_init_value=qs,
                #     delta=delta,
                #     extra_feed_dict={
                #         ph_xs: x,
                #         ph_ys: y,
                #         ph_zs: z,
                #         ph_mol_idxs: mol_idxs,
                #         ph_atom_types: atom_types
                #     }
                # )

                # print(prec, "dq", error, tol)
                # assert error < tol


                # test featurization
                grad_op = tf.gradients(f0, [ph_xs, ph_ys, ph_zs])

                grads = self.sess.run(grad_op, feed_dict={
                    ph_xs: x,
                    ph_ys: y,
                    ph_zs: z,
                    ph_mol_idxs: mol_idxs,
                    ph_atom_types: atom_types
                })

                if prec == tf.float32:
                    delta = 1e-3
                    tol = 1e-3
                elif prec == tf.float64:
                    delta = 1e-6
                    tol = 1e-9
                else:
                    raise Exception("Unknown precision")

                error = tf.test.compute_gradient_error(
                    ph_xs,
                    x.shape,
                    features,
                    (11, FEATURE_SIZE),
                    x_init_value=x,
                    delta=delta,
                    extra_feed_dict={
                        ph_ys: y,
                        ph_zs: z,
                        ph_mol_idxs: mol_idxs,
                        ph_atom_types: atom_types
                    }
                )

                print(prec, "dx", error, tol)
                assert error < tol


                error = tf.test.compute_gradient_error(
                    ph_ys,
                    y.shape,
                    features,
                    (11, FEATURE_SIZE),
                    x_init_value=y,
                    delta=delta,
                    extra_feed_dict={
                        ph_xs: x,
                        ph_zs: z,
                        ph_mol_idxs: mol_idxs,
                        ph_atom_types: atom_types
                    }
                )

                print(prec, "dy", error, tol)
                assert error < tol

                error = tf.test.compute_gradient_error(
                    ph_zs,
                    z.shape,
                    features,
                    (11, FEATURE_SIZE),
                    x_init_value=z,
                    delta=delta,
                    extra_feed_dict={
                        ph_xs: x,
                        ph_ys: y,
                        ph_mol_idxs: mol_idxs,
                        ph_atom_types: atom_types
                    }
                )

                print(prec, "dz", error, tol)
                assert error < tol




if __name__ == "__main__":
    unittest.main()
