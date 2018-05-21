import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.opt import NadamOptimizer

import khan
from khan.utils.helpers import ed_harder_rmse
from khan.model.nn import MoleculeNN, mnn_staging
from data_utils import HARTREE_TO_KCAL_PER_MOL

ani_mod = None

@ops.RegisterGradient("Featurize")
def _feat_grad(op, grad_hs, grad_cs, grad_ns, grad_os):
    x,y,z,a,mo,macs,sis,acs = op.inputs
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
        grad_os
    )

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


def initialize_module(so_file):
    global ani_mod
    if ani_mod is not None:
        raise Exception("Module has already been initialized")
    ani_mod = tf.load_op_library(so_file)


def flatten_results(res, pos=0):
    flattened = []
    for l in res:
        flattened.append(l[pos])
    return np.concatenate(flattened).reshape((-1,))


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.

    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:

            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


class TrainerMultiTower():

    def __init__(self,
        sess,
        towers,
        layer_sizes=(128, 128, 64, 8, 1),
        fit_charges=False,
        charge_layer_sizes=(64, 64, 8, 1)
    ):
        """
        A queue-enabled multi-gpu trainer. Construction of this class will also
        finalize and initialize all the variables pertaining to the input session.

        Parameters
        ----------
        sess: tf.Session
            A tensorflow session under which we use
        """
        self.towers = towers
        self.num_towers = len(towers)

        assert self.num_towers > 0

        self.x_enq = tf.placeholder(dtype=tf.float32)
        self.y_enq = tf.placeholder(dtype=tf.float32)
        self.z_enq = tf.placeholder(dtype=tf.float32)
        self.a_enq = tf.placeholder(dtype=tf.int32)
        self.m_enq = tf.placeholder(dtype=tf.int32)
        self.yt_enq = tf.placeholder(dtype=tf.float32)
        self.bi_enq = tf.placeholder(dtype=tf.int32)

        queue = tf.FIFOQueue(capacity=20*self.num_towers, dtypes=[
                tf.float32,  # Xs
                tf.float32,  # Ys
                tf.float32,  # Zs
                tf.int32,    # As
                tf.int32,    # mol ids
                tf.float32,  # Y TRUEss
                tf.int32,    # b_idxs
            ]);

        self.put_op = queue.enqueue([
            self.x_enq,
            self.y_enq,
            self.z_enq,
            self.a_enq,
            self.m_enq,
            self.yt_enq,
            self.bi_enq
        ])

        self.sess = sess

        with tf.device('/cpu:0'):
            self.learning_rate = tf.get_variable('learning_rate', tuple(), tf.float32, tf.constant_initializer(1e-4), trainable=False)
            self.optimizer = NadamOptimizer( # tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-8) # default is 1e-8
            #self.learning_rate = tf.get_variable('learning_rate', tuple(), tf.float32, tf.constant_initializer(0.1), trainable=False)
            #self.optimizer = PowerSignOptimizer(learning_rate=self.learning_rate)

            self.global_step = tf.get_variable('global_step', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
            self.decr_learning_rate = tf.assign(self.learning_rate, tf.multiply(self.learning_rate, 0.5))
            self.global_epoch_count = tf.get_variable('global_epoch_count', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
            self.local_epoch_count = tf.get_variable('local_epoch_count', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
            self.incr_global_epoch_count = tf.assign(self.global_epoch_count, tf.add(self.global_epoch_count, 1))
            self.incr_local_epoch_count = tf.assign(self.local_epoch_count, tf.add(self.local_epoch_count, 1))
            self.reset_local_epoch_count = tf.assign(self.local_epoch_count, 0)

            # these data elements are unordered since the gpu grabs the batches in different orders
            self.tower_grads = [] # average is order invariant
            self.tower_preds = []
            self.tower_bids = []
            self.tower_l2s = []
            self.tower_mos = []
            self.tower_coord_grads = []
            self.tower_features = []
            self.all_models = []
            self.tower_exp_loss = []

            with tf.variable_scope(tf.get_variable_scope()):
                # for gpu_idx in range(self.num_gpus):
                for tower_idx, tower_device in enumerate(towers):
                    with tf.device(tower_device):
                        with tf.name_scope("%s_%d" % ("tower", tower_idx)) as scope:

                            with tf.device('/cpu:0'):
                                get_op = queue.dequeue()
                                x_deq, y_deq, z_deq, a_deq, m_deq, labels, bi_deq = get_op[0], get_op[1], get_op[2], get_op[3], get_op[4], get_op[5], get_op[6]
                                self.tower_bids.append(bi_deq)
                                mol_atom_counts = tf.segment_sum(tf.ones_like(m_deq), m_deq)
                                mol_offsets = tf.cumsum(mol_atom_counts, exclusive=True)

                                scatter_idxs, gather_idxs, atom_counts = ani_mod.ani_sort(a_deq)

                                self.tower_mos.append(mol_offsets)

                            with tf.device(tower_device):
                                f0, f1, f2, f3 = ani_mod.featurize(
                                    x_deq,
                                    y_deq,
                                    z_deq,
                                    a_deq,
                                    mol_offsets,
                                    mol_atom_counts,
                                    scatter_idxs,
                                    atom_counts,
                                    name="ani_op_"+str(tower_idx)
                                )
                                feat_size = f0.op.get_attr("feature_size")

                                # TODO: optimize in C++ code directly to avoid reshape
                                f0 = tf.reshape(f0, (-1, feat_size))
                                f1 = tf.reshape(f1, (-1, feat_size))
                                f2 = tf.reshape(f2, (-1, feat_size))
                                f3 = tf.reshape(f3, (-1, feat_size))

                            self.tower_features.append(tf.gather(
                                tf.concat([f0, f1, f2, f3], axis=0),
                                gather_idxs
                            ))

                            tower_model_near = MoleculeNN(
                                type_map=["H", "C", "N", "O"],
                                atom_type_features=[f0, f1, f2, f3],
                                gather_idxs=gather_idxs,
                                layer_sizes=(feat_size,) + layer_sizes,
                                prefix="near_")

                            self.all_models.append(tower_model_near)
                            tower_near_energy = tf.segment_sum(tower_model_near.atom_outputs, m_deq)
                            if fit_charges:
                                tower_model_charges = MoleculeNN(
                                    type_map=["H", "C", "N", "O"],
                                    atom_type_features=[f0, f1, f2, f3],
                                    gather_idxs=gather_idxs,
                                    layer_sizes=(feat_size,) + charge_layer_sizes,
                                    prefix="charge_")

                                self.all_models.append(tower_model_charges)
                                tower_charges = tower_model_charges.atom_outputs

                                # (ytz + stevenso): we want to normalize the compute the charge per molecule
                                # note that this only works for *neutral* molecules. For molecules that have a formal charge
                                # we want to specify correct differently, or turn off the normalization entirely.
                                tower_charges_per_mol = tf.segment_sum(tower_charges, m_deq) # per molecule charge
                                tower_charges_per_mol = tf.divide(tower_charges_per_mol, tf.cast(mol_atom_counts, dtype=tf.float32)) # per molecule avg charge
                                tower_charge_correction = tf.gather(tower_charges_per_mol, m_deq) # generate the per atom correction
                                tower_charges = tf.subtract(tower_charges, tower_charge_correction) # zero out the charge

                                tower_far_energy = ani_mod.ani_charge(
                                    x_deq,
                                    y_deq,
                                    z_deq,
                                    tower_charges,
                                    mol_offsets,
                                    mol_atom_counts
                                )
                                tower_pred = tf.add(tower_near_energy, tower_far_energy)
                            else:
                                tower_pred = tower_near_energy

                            self.tower_preds.append(tower_pred)
                            tower_l2 = tf.squared_difference(tower_pred, labels)
                            self.tower_l2s.append(tower_l2)

                            tower_rmse = tf.sqrt(tf.reduce_mean(tower_l2))
                            tower_exp_loss = tf.exp(tf.cast(tower_rmse, dtype=tf.float64))

                            tf.get_variable_scope().reuse_variables()
                            self.tower_exp_loss.append(tower_exp_loss)
                            tower_grad = self.optimizer.compute_gradients(tower_exp_loss)
                            self.tower_grads.append(tower_grad)

                            self.tower_coord_grads.append(tf.gradients(tower_pred, [x_deq, y_deq, z_deq]))

            self.best_params = []
            self.save_best_params_ops = []
            self.load_best_params_ops = []

            for var in tf.trainable_variables():
                copy_name = "best_"+var.name.split(":")[0]
                copy_shape = var.shape
                copy_type = var.dtype
                var_copy = tf.get_variable(copy_name, copy_shape, copy_type, tf.zeros_initializer, trainable=False)
                self.best_params.append(var_copy)
                self.save_best_params_ops.append(tf.assign(var_copy, var))
                self.load_best_params_ops.append(tf.assign(var, var_copy))

            apply_gradient_op = self.optimizer.apply_gradients(average_gradients(self.tower_grads), global_step=self.global_step)
            variable_averages = tf.train.ExponentialMovingAverage(0.9999, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            self.train_op = tf.group(apply_gradient_op, variables_averages_op)

        ws = self._weight_matrices()
        max_norm_ops = []
        for w in ws:
            max_norm_ops.append(tf.assign(w, tf.clip_by_norm(w, 3.0, axes=1)))
        self.max_norm_ops = max_norm_ops

        self.unordered_l2s = tf.squeeze(tf.concat(self.tower_l2s, axis=0))
        #self.unordered_l2s += l2_norm_k * tf.norm(ws) # one way to enforce an l2 norm

        self.global_initializer_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def initialize(self):
        """
        Randomly initialize the parameters in the trainer's underlying model.
        """
        self.sess.run(self.global_initializer_op)

    def save_best_params(self):
        """
        Copy the current model's trainable parameters as the best so far.
        """
        self.sess.run(self.save_best_params_ops)

    def load_best_params(self):
        """
        Restore the current model's parameters from the best found so far.
        """
        self.sess.run(self.load_best_params_ops)

    def save(self, save_dir):
        """
        Save the entire model to a given directory.

        Parameters
        ----------
        save_dir: str
            Path of the save_dir. If the path does not exist then it will
            be created automatically.

        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "model.ckpt")
        self.saver.save(self.sess, save_path)

    def load(self, save_dir):
        """
        Load an existing model from an existing directory and initialize
        the trainer's Session variables.

        Parameters
        ----------
        save_dir: str
            Directory containing the checkpoint file. This should be the same
            as what was passed into save().

        .. note:: It is expected that this directory exists.

        """
        save_path = os.path.join(save_dir, "model.ckpt")
        self.saver.restore(self.sess, save_path)

    def _weight_matrices(self):
        weights = []
        # vars are always shared so we can just grab them by the first tower
        for ann in self.all_models[0].anns:
            for W in ann.Ws:
                weights.append(W)
        return weights

    def _biases(self):
        biases = []
        for ann in self.all_models[0].anns:
            for b in ann.bs:
                biases.append(b)
        return biases

    def get_train_op_rmse(self):
        return self.train_op_rmse

    def get_train_op_exp(self):
        return self.train_op_exp

    def get_loss_op(self):
        return self.exp_loss

    def eval_abs_rmse(self, dataset, batch_size=1024):
        test_l2s = self.feed_dataset(dataset, shuffle=False, target_ops=[self.unordered_l2s], batch_size=batch_size)
        return np.sqrt(np.mean(flatten_results(test_l2s))) * HARTREE_TO_KCAL_PER_MOL

    def coordinate_gradients(self, dataset, batch_size=1024):
        """
        Compute gradients for a dataset.

        Parameters
        ----------
        dataset: khan.RawDataset
            Dataset from which we predict from.

        batch_size: int (optional)
            Size of each batch used during prediction.

        Returns
        -------
        list of gradients
            Returns a list of num_atoms x 3 gradients for each molecule

        """
        results = self.feed_dataset(dataset, shuffle=False, target_ops=[self.tower_coord_grads, self.tower_mos, self.tower_bids], batch_size=batch_size)

        for (grad, mo, tids) in results:
            bidxs = np.argsort(tids)
            sorted_grads = np.take(grad, bidxs, axis=0)
            sorted_mos = np.take(mo, bidxs, axis=0)

            for (mo, grad_all) in zip(sorted_mos, sorted_grads):

                grad_x, grad_y, grad_z = grad_all
                grad_xs = np.split(grad_x, mo)
                grad_ys = np.split(grad_y, mo)
                grad_zs = np.split(grad_z, mo)

                for x,y,z in zip(grad_xs, grad_ys, grad_zs):
                    grad_xyz = np.vstack([x,y,z]).transpose()
                    yield grad_xyz


    def featurize(self, dataset, batch_size=1024):
        """
        Featurize a given dataset.

        Parameters
        ----------
        dataset: khan.RawDataset
            Dataset used for featurization.

        batch_size: int (optional)
            Size of each batch.

        Returns
        -------
        list of np.ndarray
            Returns a list of numpy array corresponding to the features
            of each molecule in the dataset.

        """

        results = self.feed_dataset(dataset, shuffle=False, target_ops=[self.tower_features, self.tower_mos, self.tower_bids], batch_size=batch_size)

        for (feats, mos, tids) in results:
            bidxs = np.argsort(tids)
            sorted_mos = np.take(mos, bidxs, axis=0)
            sorted_feats = np.take(feats, bidxs, axis=0)

            for (mo, feat) in zip(sorted_mos, sorted_feats):
                feats = np.split(feat, mo)
                for f in feats:
                    yield f


    def predict(self, dataset, batch_size=1024):
        """
        Infer y-values given a dataset.

        Parameters
        ----------
        dataset: khan.RawDataset
            Dataset from which we predict from.

        batch_size: int (optional)
            Size of each batch used during prediction.

        Returns
        -------
        list of floats
            Returns a list of predicted [y0, y1, y2...] in the same order as the dataset Xs [x0, x1, x2...]

        """
        results = self.feed_dataset(dataset, shuffle=False, target_ops=[self.tower_preds, self.tower_bids], batch_size=batch_size)
        ordered_ys = []
        for (ys, ids) in results:
            sorted_ys = np.take(ys, np.argsort(ids), axis=0)
            ordered_ys.extend(np.concatenate(sorted_ys, axis=0))
        return ordered_ys

    def eval_eh_rmse(self, dataset, group_ys, batch_size=1024):
        ordered_ys = self.predict(dataset, batch_size)
        return ed_harder_rmse(group_ys, ordered_ys) * HARTREE_TO_KCAL_PER_MOL

    def feed_dataset(self,
        dataset,
        shuffle,
        target_ops,
        batch_size,
        before_hooks=None):
        """
        Feed a dataset into the trainer under arbitrary ops.

        Params
        ------
        dataset: khan.RawDataset
            Input dataset that may or may not have y-values depending on the target_ops

        shuffle: bool
            Whether or not we randomly shuffle the data before feeding.

        target_ops: list of tf.Tensors
            tf.Tensors for which we wish to obtain values for.

        batch_size: int
            Size of the batch for which we iterate the dataset over.

        hooks: list of tf.Ops
            List of tensorflow ops which we run before every batch. Note that currently
            these ops must have no feed_dict dependency.

        Returns
        -------
        A generator that yields results of the specified op in increments of batch_size.

        .. note:: You must ensure that resulting generator is fully iterated over to ensure
        proper terminating of the submission threads. Furthermore, the resulting iterable
        should be as non-blocking as possible, since flushing of the queue assumes that the
        results are consumed asap.

        """

        def submitter():

            accum = 0
            g_b_idx = 0

            # for i in range(self.num_gpus):

            # suppose we have 4 gpus and 5 batches
            # the distribution schedule is as follows:
            # gpu   0 1 2 3
            # bid0  1 1 1 1
            # bid1  1 0 0 0

            # suppose we have 3 gpus and 5 batches
            # the distribution schedule is as follows:
            # gpu   0 1 2
            # bid0  1 1 1
            # bid1  1 1 0

            try:

                n_batches = dataset.num_batches(batch_size)

                for b_idx, (mol_xs, mol_idxs, mol_yts) in enumerate(dataset.iterate(batch_size=batch_size, shuffle=shuffle)):
                    atom_types = (mol_xs[:, 0]).astype(np.int32)
                    if before_hooks:
                        self.sess.run(before_hooks)
                    self.sess.run(self.put_op, feed_dict={
                        self.x_enq: mol_xs[:, 1],
                        self.y_enq: mol_xs[:, 2],
                        self.z_enq: mol_xs[:, 3],
                        self.a_enq: atom_types,
                        self.m_enq: mol_idxs,
                        self.yt_enq: mol_yts,
                        self.bi_enq: b_idx
                    })

                    g_b_idx += 1

                remainder = n_batches % self.num_towers
                if remainder:
                    for _ in range(self.num_towers - remainder):
                        if before_hooks:
                            self.sess.run(before_hooks)
                        self.sess.run(self.put_op, feed_dict={
                            self.x_enq: np.zeros((0, 1), dtype=np.float32),
                            self.y_enq: np.zeros((0, 1), dtype=np.float32),
                            self.z_enq: np.zeros((0, 1), dtype=np.float32),
                            self.a_enq: np.zeros((0, ), dtype=np.int32),
                            self.m_enq: np.zeros((0, ), dtype=np.int32),
                            self.yt_enq: np.zeros((0, )),
                            self.bi_enq: b_idx,
                        })
                        b_idx += 1
            except Exception as e:
                print("QueueError:", e)

        executor = ThreadPoolExecutor(4)
        executor.submit(submitter)

        n_tower_batches = -(-dataset.num_batches(batch_size=batch_size) // self.num_towers)

        for i in range(n_tower_batches):
            yield self.sess.run(target_ops)
