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


class TrainerMultiGPU():

    # Note: this is a pretty terrible class all in all. I'm ashamed of writing it
    # but it gets the job done.


    def set_session(self, sess):
        self.sess = sess

    def initialize(self):
        # todo: call by default
        self.sess.run(self.global_initializer_op)

    def save_best_params(self):
        self.sess.run(self.save_best_params_ops)

    def load_best_params(self):
        self.sess.run(self.load_best_params_ops)

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "model.ckpt")
        self.saver.save(self.sess, save_path)


    def load(self, save_dir):
        save_path = os.path.join(save_dir, "model.ckpt")
        self.saver.restore(self.sess, save_path)

    def __init__(
        self,
        sess,
        f0_debug=None,
        f1_debug=None,
        f2_debug=None,
        f3_debug=None):

        self.num_gpus = 3

        self.x_enq = tf.placeholder(dtype=tf.float32)
        self.y_enq = tf.placeholder(dtype=tf.float32)
        self.z_enq = tf.placeholder(dtype=tf.float32)
        self.a_enq = tf.placeholder(dtype=tf.int32)
        self.m_enq = tf.placeholder(dtype=tf.int32)
        self.yt_enq = tf.placeholder(dtype=tf.float32)

        queue = tf.FIFOQueue(capacity=50, dtypes=[
                tf.float32,  # Xs
                tf.float32,  # Ys
                tf.float32,  # Zs
                tf.int32,    # As
                tf.int32,    # mol ids
                tf.float32   # Y TRUEss
            ]);

        self.put_op = queue.enqueue([
            self.x_enq,
            self.y_enq,
            self.z_enq,
            self.a_enq,
            self.m_enq,
            self.yt_enq
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

            self.local_epoch_count = tf.get_variable('local_epoch_count', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)

            self.incr_local_epoch_count = tf.assign(self.local_epoch_count, tf.add(self.local_epoch_count, 1))
            self.reset_local_epoch_count = tf.assign(self.local_epoch_count, 0)

            self.all_grads = [] # do something similar for all L2s, etc.
            self.all_preds = []
            self.all_l2s = []
            self.all_models = []
            self.tower_exp_loss = []
            self.tower_grads = []

            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope("%s_%d" % ("tower", i)) as scope:

                            with tf.device('/cpu:0'):

                                get_op = queue.dequeue()
                                x_deq, y_deq, z_deq, a_deq, m_deq, labels = get_op[0], get_op[1], get_op[2], get_op[3], get_op[4], get_op[5]
                                mol_atom_counts = tf.segment_sum(tf.ones_like(m_deq), m_deq)
                                mol_offsets = tf.cumsum(mol_atom_counts, exclusive=True)
                                scatter_idxs, gather_idxs, atom_counts = sort_lib.ani_sort(a_deq)

                            with tf.device('/gpu:%d' % i):

                                f0, f1, f2, f3 = ani_mod.ani(
                                    x_deq,
                                    y_deq,
                                    z_deq,
                                    a_deq,
                                    mol_offsets,
                                    mol_atom_counts,
                                    scatter_idxs,
                                    atom_counts
                                )

                                # TODO: optimize in C++ code directly to avoid reshape
                                f0 = tf.reshape(f0, (-1, 384))
                                f1 = tf.reshape(f1, (-1, 384))
                                f2 = tf.reshape(f2, (-1, 384))
                                f3 = tf.reshape(f3, (-1, 384))

                            tower_model = MoleculeNN(
                                type_map=["H", "C", "N", "O"],
                                atom_type_features=[f0, f1, f2, f3],
                                gather_idxs=gather_idxs,
                                mol_idxs=m_deq,
                                layer_sizes=(384, 256, 128, 64, 1))

                            self.all_models.append(tower_model)

                            tower_pred = tower_model.predict_op()
                            self.all_preds.append(tower_pred)
                            tower_l2 = tf.squared_difference(tower_pred, labels)
                            self.all_l2s.append(tower_l2)

                            tower_rmse = tf.sqrt(tf.reduce_mean(tower_l2))
                            tower_exp_loss = tf.exp(tf.cast(tower_rmse, dtype=tf.float64))

                            tf.get_variable_scope().reuse_variables()
                            self.tower_exp_loss.append(tower_exp_loss)
                            tower_grad = self.optimizer.compute_gradients(tower_exp_loss)
                            self.all_grads.append(tower_grad)

            apply_gradient_op = self.optimizer.apply_gradients(average_gradients(self.all_grads), global_step=self.global_step)
            variable_averages = tf.train.ExponentialMovingAverage(0.9999, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            self.train_op = tf.group(apply_gradient_op, variables_averages_op)

        ws = self.weight_matrices()
        max_norm_ops = []

        for w in ws:
            max_norm_ops.append(tf.assign(w, tf.clip_by_norm(w, 2.0, axes=1)))

        self.l2 = tf.squeeze(tf.concat(self.all_l2s, axis=0))
        self.preds = tf.squeeze(tf.concat(self.all_preds, axis=0))

        self.max_norm_ops = max_norm_ops
        # TODO: MAX_NORM
 
        # self.x_enq = x_enq
        # self.y_enq = y_enq
        # self.z_enq = z_enq
        # self.a_enq = a_enq
        # self.m_enq = m_enq
        # self.si_enq = si_enq
        # self.gi_enq = gi_enq
        # self.ac_enq = ac_enq
        # self.yt_enq = yt_enq
        # self.put_op = put_op

        # ytz: finalized - so the saver needs to be at the end when all vars have been created.
        # (though not necessarily initialized)

        self.global_initializer_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
 

        return


    def weight_matrices(self):
        weights = []
        # vars are shared so we just grab them by the first tower
        for ann in self.all_models[0].anns:
            for W in ann.Ws:
                weights.append(W)
        return weights

    def biases(self):
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
        test_l2s = self.feed_dataset(dataset, shuffle=False, target_ops=[self.l2], batch_size=batch_size)
        return np.sqrt(np.mean(flatten_results(test_l2s))) * HARTREE_TO_KCAL_PER_MOL

    def eval_eh_rmse(self, dataset, group_ys, batch_size=1024):
        ys = self.feed_dataset(dataset, shuffle=False, target_ops=[self.preds], batch_size=batch_size)
        return ed_harder_rmse(group_ys, flatten_results(ys)) * HARTREE_TO_KCAL_PER_MOL

    def feed_dataset(self,
        dataset,
        shuffle,
        target_ops,
        batch_size):

        # batch_size = 1024

        st = time.time()

        # print("num_batches:", dataset.num_batches(batch_size=batch_size))

        def submitter():

            accum = 0
            g_b_idx = 0

            for i in range(self.num_gpus):
                for b_idx, (mol_xs, mol_idxs, mol_yts) in enumerate(dataset.iterate_advanced(batch_size=batch_size, shuffle=shuffle)): # FIX SHUFFLE TO SHUFFLE
                    atom_types = (mol_xs[:, 0]).astype(np.int32)
                    try:
                        # print("feeding...", g_b_idx)
                        self.sess.run(self.put_op, feed_dict={
                            self.x_enq: mol_xs[:, 1],
                            self.y_enq: mol_xs[:, 2],
                            self.z_enq: mol_xs[:, 3],
                            self.a_enq: atom_types,
                            self.m_enq: mol_idxs,
                            self.yt_enq: mol_yts,
                        })
                        g_b_idx += 1
                    except Exception as e:
                        print("EEEEE", e)

        executor = ThreadPoolExecutor(4)
        executor.submit(submitter)

        results = []
        n_batches = dataset.num_batches(batch_size=batch_size)
        for i in range(n_batches):
            res = self.sess.run(target_ops)

            # print(res)
            # if len(target_ops) == 5:
                # print("batch rmse:", np.sqrt(np.mean(res[3])))

            results.append(res)


        return results
