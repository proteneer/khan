import tensorflow as tf

class Trainer():

    def __init__(self, mnn, predictions):
        # trains an ANI-1 mnn.
        self.mnn = mnn
        self.preds = predictions
        self.tau = 0.5
        self.rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.mnn.molecule_energies(), predictions)))
        self.exp_loss = self.tau * tf.exp(self.rmse /self.tau) 
        self.optimizer = tf.train.AdamOptimizer() # change defaults
        # todo: max norm
        self.train_op = self.optimizer.minimize(self.exp_loss)

    def train(self):
        return self.train_op