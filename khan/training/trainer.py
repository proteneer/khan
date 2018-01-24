import tensorflow as tf
import khan

class Trainer():

    def __init__(
        self,
        model, # atom type scatter idxs
        labels):
        """
        Model for full end-to-end.
        

        """
        self.model = model
        self.l2 = tf.squared_difference(self.model.predict_op(), labels)
        self.rmse = tf.sqrt(tf.reduce_mean(self.l2))
        # float64 is for numerical stability

        self.exp_loss = tf.exp(tf.cast(self.rmse, dtype=tf.float64))
        self.optimizer_rmse = tf.train.AdamOptimizer() # change defaults
        # todo: add a step for max normalization
        self.train_op_rmse = self.optimizer_rmse.minimize(self.rmse)
        self.optimizer_exp = tf.train.AdamOptimizer() # change defaults
        self.train_op_exp = self.optimizer_exp.minimize(self.exp_loss)

        # maxnorm
        ws = self.weight_matrices()
        max_norm_ops = []

        for w in ws:
            max_norm_ops.append(tf.assign(w, tf.clip_by_norm(w, 3.0, axes=1)))

        self.max_norm_ops = max_norm_ops

    def weight_matrices(self):
        weights = []
        for ann in self.model.anns:
            for W in ann.Ws:
                weights.append(W)
        return weights

    def get_maxnorm_ops(self):
        return self.max_norm_ops

    def get_train_op_rmse(self):
        return self.train_op_rmse

    def get_train_op_exp(self):
        return self.train_op_exp

    def get_loss_op(self):
        return self.exp_loss

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
