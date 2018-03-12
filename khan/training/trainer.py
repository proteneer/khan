import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf

import khan
from khan.utils.helpers import ed_harder_rmse
from khan.model.nn import MoleculeNN, mnn_staging

def flatten_results(res, pos=0):
    flattened = []
    for l in res:
        flattened.append(l[pos])
    return np.concatenate(flattened).reshape((-1,))

class Trainer():

    # Note: this is a pretty terrible class all in all. I'm ashamed of writing it
    # but it gets the job done.

    @classmethod
    def from_mnn_queue(
        cls,
        session):
        (f0_enq, f1_enq, f2_enq, f3_enq, gi_enq, mi_enq, yt_enq), \
        (f0_deq, f1_deq, f2_deq, f3_deq, gi_deq, mi_deq, yt_deq), \
        put_op = mnn_staging()

        mnn = MoleculeNN(
            type_map=["H", "C", "N", "O"],
            atom_type_features=[f0_deq, f1_deq, f2_deq, f3_deq],
            gather_idxs=gi_deq,
            mol_idxs=mi_deq,
            layer_sizes=(384, 256, 128, 64, 1))

        return cls(
            mnn,
            yt_deq,
            f0_enq,
            f1_enq,
            f2_enq,
            f3_enq,
            gi_enq,
            mi_enq,
            yt_enq,
            put_op,
            session,
        )

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def save(self, save_dir):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_prefix = "ani"
        save_path = os.path.join(save_dir, save_prefix)

        # with tf.sess:
            # print

        save_path = self.saver.save(self.sess, save_path, global_step=self.global_step)

        # print(tf.train)

        # print("Saving to...", save_path)

    def load(self, save_dir):
        save_prefix = "ani"
        save_path = os.path.join(save_dir, save_prefix)
        checkpoints = glob.glob(save_path+"*.index")
        max_gstep = 0
        for f in checkpoints:
            g_step = int(f.split('.')[0].split("-")[1])
            if g_step > max_gstep:
                max_gstep = g_step

        last_file = save_path+"-"+str(max_gstep)
        print("loading from", last_file)
        self.saver.restore(self.sess, last_file)

    def __init__(
        self,
        model, # atom type scatter idxs
        labels,
        f0_enq,
        f1_enq,
        f2_enq,
        f3_enq,
        gi_enq,
        mi_enq,
        yt_enq,
        put_op,
        sess
        ):
        """
        Model for full end-to-end.
        """

        self.sess = sess

        self.model = model
        self.l2 = tf.squared_difference(self.model.predict_op(), labels)
        self.rmse = tf.sqrt(tf.reduce_mean(self.l2))

        self.exp_loss = tf.exp(tf.cast(self.rmse, dtype=tf.float64))
        self.global_step = tf.get_variable('global_step', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.get_variable('learning_rate', tuple(), tf.float32, tf.constant_initializer(0.001), trainable=False)
        # self.optimizer_rmse = tf.train.AdamOptimizer(
        #     learning_rate=self.learning_rate,
        #     beta1=0.9,
        #     beta2=0.999,
        #     epsilon=1e-8) # change defaults
        # # todo: add a step for max normalization
        # self.train_op_rmse = self.optimizer_rmse.minimize(self.rmse)
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

        self.f0_enq = f0_enq
        self.f1_enq = f1_enq
        self.f2_enq = f2_enq
        self.f3_enq = f3_enq
        self.gi_enq = gi_enq
        self.mi_enq = mi_enq
        self.yt_enq = yt_enq
        self.put_op = put_op


        # ytz: finalized - so the saver needs to be at the end when all vars have been created.
        # (though not necessarily initialized)
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

    def eval_abs_rmse(self, dataset):
        test_l2s = self.feed_dataset(dataset, shuffle=False, target_ops=[self.l2])
        return np.sqrt(np.mean(flatten_results(test_l2s)))

    def eval_eh_rmse(self, dataset, group_ys):
        ys = self.feed_dataset(dataset, shuffle=False, target_ops=[self.model.predict_op()])
        return ed_harder_rmse(group_ys, flatten_results(ys))

    def feed_dataset(self,
        dataset,
        shuffle,
        target_ops):

        def submitter():
            for b_idx, (f0, f1, f2, f3, gi, mi, yt) in enumerate(dataset.iterate(shuffle=shuffle)):
                try:
                    self.sess.run(self.put_op, feed_dict={
                        self.f0_enq: f0,
                        self.f1_enq: f1,
                        self.f2_enq: f2,
                        self.f3_enq: f3,
                        self.gi_enq: gi,
                        self.mi_enq: mi,
                        self.yt_enq: yt,
                    })
                except Exception as e:
                    print("EEEEE", e)

        executor = ThreadPoolExecutor(4)
        executor.submit(submitter)

        results = []

        for i in range(dataset.num_batches()):
            res = self.sess.run(target_ops)
            results.append(res)

        return results
