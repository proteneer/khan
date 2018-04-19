import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import time

ani_mod = tf.load_op_library('gpu_featurizer/ani.so');
sort_lib = tf.load_op_library('gpu_featurizer/ani_sort.so');


import khan
from khan.utils.helpers import ed_harder_rmse
from khan.model.nn import MoleculeNN, mnn_staging

from data_utils import HARTREE_TO_KCAL_PER_MOL


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
        layer_sizes=(128, 128, 64, 1)
    ):
        """
        A queue-enabled multi-gpu trainer. Construction of this class will also
        finalize and initialize all the variables pertaining to the input session.

        Parameters
        ----------
        sess: tf.Session
            A tensorflow session under which we use

        n_gpus: int (optional)
            Number of gpus to train the model on. This must be > 0 for now.

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

            self.learning_rate = tf.get_variable('learning_rate', tuple(), tf.float32, tf.constant_initializer(1e-3), trainable=False)
            self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-8) # change defaults

            self.global_step = tf.get_variable('global_step', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
            self.decr_learning_rate = tf.assign(self.learning_rate, tf.multiply(self.learning_rate, 0.1))
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
                                scatter_idxs, gather_idxs, atom_counts = sort_lib.ani_sort(a_deq)

                            with tf.device(tower_device):
                                f0, f1, f2, f3 = ani_mod.ani(
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

                            tower_model = MoleculeNN(
                                type_map=["H", "C", "N", "O"],
                                atom_type_features=[f0, f1, f2, f3],
                                gather_idxs=gather_idxs,
                                mol_idxs=m_deq,
                                layer_sizes=(feat_size,) + layer_sizes)

                            self.all_models.append(tower_model)

                            tower_pred = tower_model.predict_op()
                            self.tower_preds.append(tower_pred)
                            tower_l2 = tf.squared_difference(tower_pred, labels)
                            self.tower_l2s.append(tower_l2)

                            tower_rmse = tf.sqrt(tf.reduce_mean(tower_l2))
                            tower_exp_loss = tf.exp(tf.cast(tower_rmse, dtype=tf.float64))

                            tf.get_variable_scope().reuse_variables()
                            self.tower_exp_loss.append(tower_exp_loss)
                            tower_grad = self.optimizer.compute_gradients(tower_exp_loss)
                            self.tower_grads.append(tower_grad)

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

        self.unordered_l2s = tf.squeeze(tf.concat(self.tower_l2s, axis=0))
        self.max_norm_ops = max_norm_ops

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

    def get_maxnorm_ops(self):
        return self.max_norm_ops

    def get_train_op_rmse(self):
        return self.train_op_rmse

    def get_train_op_exp(self):
        return self.train_op_exp

    def get_loss_op(self):
        return self.exp_loss

    def eval_abs_rmse(self, dataset, batch_size=1024):
        test_l2s = self.feed_dataset(dataset, shuffle=False, target_ops=[self.unordered_l2s], batch_size=batch_size)
        return np.sqrt(np.mean(flatten_results(test_l2s))) * HARTREE_TO_KCAL_PER_MOL

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
        batch_size):
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

        results = []
        n_tower_batches = -(-dataset.num_batches(batch_size=batch_size) // self.num_towers)

        for i in range(n_tower_batches):
            res = self.sess.run(target_ops)
            results.append(res)

        return results
