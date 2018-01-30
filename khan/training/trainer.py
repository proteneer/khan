import tensorflow as tf
import khan


from concurrent.futures import ThreadPoolExecutor

class Trainer():

    # Note: this is a pretty terrible class all in all. I'm ashamed of writing it
    # but it gets the job done.

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
        ):
        """
        Model for full end-to-end.
        

        """
        self.model = model
        self.l2 = tf.squared_difference(self.model.predict_op(), labels)
        self.rmse = tf.sqrt(tf.reduce_mean(self.l2))
        # float64 is for numerical stability

        self.exp_loss = tf.exp(tf.cast(self.rmse, dtype=tf.float64))
        self.global_step = tf.get_variable('global_step', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
        # b = tf.get_variable("b"+name, (y), np.float32, tf.zeros_initializer)
        self.learning_rate = tf.get_variable("learning_rate", tuple(), tf.float32, tf.constant_initializer(0.001), trainable=False)
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

    def feed_dataset(self,
        session,
        dataset,
        shuffle,
        target_ops):

        def submitter():
            for b_idx, (f0, f1, f2, f3, gi, mi, yt) in enumerate(dataset.iterate(shuffle=shuffle)):
                # print("submitting...", b_idx)
                try:
                    session.run(self.put_op, feed_dict={
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
            # print("running...")
            results.append(session.run(target_ops))
            # print(res)

        return results

    # def get_atm(self)

    # def get_input_ops(self, precomputed_features=False):
    #     """
    #     Get the input ops required for training.

    #     Parameters
    #     ----------
    #     precomputed_features: bool
    #         Determines which types of features we need to return.

    #     """

    #     if precomputed_features:
    #         # lsit o
    #         return self.atom_type_feats, self.gather_idxs, self.mol_idxs
    #     else:
    #         return self.bam, self.mol_offsets, self.gather_idxs, self.mol_idxs

    def get_slow_training_ops():
        pass

    def run_epoch(self):
        pass
