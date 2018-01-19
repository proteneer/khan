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
        self.tau = 0.5
        self.rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.model.predict_op(), labels)))
        self.exp_loss = self.tau * tf.exp(self.rmse /self.tau) 
        self.optimizer = tf.train.AdamOptimizer() # change defaults
        # todo: add a step for max normalization
        self.train_op = self.optimizer.minimize(self.exp_loss)

    def get_train_op(self):
        return self.train_op

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
