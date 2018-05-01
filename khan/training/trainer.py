import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import time

ani_mod = tf.load_op_library('gpu_featurizer/ani.so');


import khan
from khan.utils.helpers import ed_harder_rmse
from khan.model.nn import MoleculeNN, mnn_staging

from data_utils import HARTREE_TO_KCAL_PER_MOL

# import time
# import numpy as np
# import tensorflow as tf

# ani_mod = tf.load_op_library('ani.so');

# Xs = np.load('Xs.npy')
# Ys = np.load('Ys.npy')
# Zs = np.load('Zs.npy')
# As = np.load('As.npy')
# MOs = np.load('MOs.npy')
# MACs = np.load('MACs.npy')
# # TCs = np.zeros(4, dtype=np.int32)

# # for z in As:
# #   # print(z)
# #   TCs[z] += 1

# feat = ani_mod.ani(Xs, Ys, Zs, As, MOs, MACs)

# st = time.time()

# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

#     for idx in range(1000):
#         results = sess.run(feat)
#         print(len(results))
#         print("python samples per minute: ", (idx+1)*len(MOs)/(time.time()-st) * 60)

#     # res = res.reshape(len(Xs), 384)
#     # for f in res:
#         # print(f)

#     # print(res.shape)

class Trainer():

    # Note: this is a pretty terrible class all in all. I'm ashamed of writing it
    # but it gets the job done.

    @classmethod
    def from_mnn_queue(
        cls,
        session=None):

        (x_enq, y_enq, z_enq, a_enq, si_enq, gi_enq, m_enq, yt_enq), \
        (x_deq, y_deq, z_deq, a_deq, si_deq, gi_deq, m_deq, labels), \
        put_op = mnn_staging()


        mol_atom_counts = tf.segment_sum(tf.ones_like(m_deq), m_deq)        
        mol_offsets = tf.cumsum(mol_atom_counts, exclusive=True)

        f0, f1, f2, f3, gi = ani_mod.ani(x_deq, y_deq, z_deq, a_deq, mol_offsets, mol_atom_counts)

        f0 = tf.reshape(f0, (-1, 384))
        f1 = tf.reshape(f1, (-1, 384))
        f2 = tf.reshape(f2, (-1, 384))
        f3 = tf.reshape(f3, (-1, 384))

        mnn = MoleculeNN(
            type_map=["H", "C", "N", "O"],
            atom_type_features=[f0, f1, f2, f3],
            gather_idxs=gi,
            mol_idxs=m_deq,
            layer_sizes=(384, 256, 128, 64, 1))

        return cls(
            mnn,
            labels,
            x_enq,
            y_enq,
            z_enq,
            a_enq,
            m_enq,
            yt_enq,
            put_op,
            session,
            f0,
            f1,
            f2,
            f3
        )

    def set_session(self, sess):
        self.sess = sess

    def initialize(self):
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
        model, # atom type scatter idxs
        labels,
        x_enq,
        y_enq,
        z_enq,
        a_enq,
        m_enq,
        yt_enq,
        put_op,
        sess,
        f0_debug=None,
        f1_debug=None,
        f2_debug=None,
        f3_debug=None):
        """
        Model for full end-to-end.
        """
        self.sess = sess
        self.model = model
        self.best_params = []
        self.save_best_params_ops = []
        self.load_best_params_ops = []

        self.f0_debug = f0_debug
        self.f1_debug = f1_debug
        self.f2_debug = f2_debug
        self.f3_debug = f3_debug

        for var in tf.trainable_variables():

            copy_name = "best_"+var.name.split(":")[0]
            copy_shape = var.shape
            copy_type = var.dtype

            var_copy = tf.get_variable(copy_name, copy_shape, copy_type, tf.zeros_initializer, trainable=False)

            self.best_params.append(var_copy)
            self.save_best_params_ops.append(tf.assign(var_copy, var))
            self.load_best_params_ops.append(tf.assign(var, var_copy))

        self.l2 = tf.squared_difference(self.model.predict_op(), labels)
        self.rmse = tf.sqrt(tf.reduce_mean(self.l2))

        self.exp_loss = tf.exp(tf.cast(self.rmse, dtype=tf.float64))
        self.global_step = tf.get_variable('global_step', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.get_variable('learning_rate', tuple(), tf.float32, tf.constant_initializer(0.001), trainable=False)
        self.decr_learning_rate = tf.assign(self.learning_rate, tf.multiply(self.learning_rate, 0.1))

        self.local_epoch_count = tf.get_variable('local_epoch_count', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)

        self.incr_local_epoch_count = tf.assign(self.local_epoch_count, tf.add(self.local_epoch_count, 1))
        self.reset_local_epoch_count = tf.assign(self.local_epoch_count, 0)

        # debug
        self.W_grads = tf.gradients(self.rmse, self.weight_matrices())
        self.b_grads = tf.gradients(self.rmse, self.biases())
        # self.best_loss = tf.get_variable('best_loss', tuple(), tf.float32, tf.constant_initializer(9.99e9), trainable=False)

        self.optimizer_rmse = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8) # change defaults
        # todo: add a step for max normalization
        self.train_op_rmse = self.optimizer_rmse.minimize(self.rmse, global_step=self.global_step)
        self.optimizer_exp = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8) # change defaults
        self.train_op_exp = self.optimizer_exp.minimize(self.exp_loss, global_step=self.global_step)

        # maxnorm
        ws = self.weight_matrices()
        max_norm_ops = []

        for w in ws:
            max_norm_ops.append(tf.assign(w, tf.clip_by_norm(w, 2.0, axes=1)))

        self.max_norm_ops = max_norm_ops

        self.x_enq = x_enq
        self.y_enq = y_enq
        self.z_enq = z_enq
        self.a_enq = a_enq
        self.m_enq = m_enq
        self.yt_enq = yt_enq
        self.put_op = put_op

        # ytz: finalized - so the saver needs to be at the end when all vars have been created.
        # (though not necessarily initialized)

        self.global_initializer_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        


    def weight_matrices(self):
        weights = []
        for ann in self.model.anns:
            for W in ann.Ws:
                weights.append(W)
        return weights

    def biases(self):
        biases = []
        for ann in self.model.anns:
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
        ys = self.feed_dataset(dataset, shuffle=False, target_ops=[self.model.predict_op()], batch_size=batch_size)
        return ed_harder_rmse(group_ys, flatten_results(ys)) * HARTREE_TO_KCAL_PER_MOL

    def feed_dataset(self,
        dataset,
        shuffle,
        target_ops,
        batch_size):

        # batch_size = 1024

        st = time.time()

        print("num_batches:", dataset.num_batches(batch_size=batch_size))

        def submitter():
            for b_idx, (mol_xs, mol_idxs, mol_yts) in enumerate(dataset.iterate_advanced(batch_size=batch_size, shuffle=shuffle)): # FIX SHUFFLE TO SHUFFLE

                try:
                    # print("feeding...", b_idx)
                    self.sess.run(self.put_op, feed_dict={
                        self.x_enq: mol_xs[:, 1],
                        self.y_enq: mol_xs[:, 2],
                        self.z_enq: mol_xs[:, 3],
                        self.a_enq: mol_xs[:, 0].astype(np.int32),
                        self.m_enq: mol_idxs,
                        self.yt_enq: mol_yts,
                    })
                except Exception as e:
                    print("EEEEE", e)

        executor = ThreadPoolExecutor(4)
        executor.submit(submitter)

        results = []

        for i in range(dataset.num_batches(batch_size=batch_size)):
            # print("running...", i)
            res = self.sess.run(target_ops)
            
            # print("samples_per_minute:", ((i+1)*batch_size)/(time.time()-st) * 60)
            results.append(res)


        return results
