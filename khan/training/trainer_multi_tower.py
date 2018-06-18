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
from khan.data.dataset import RawDataset
from khan.model import activations

ani_mod = None



# dE/dx = (dE/df)*(df/dx)
@ops.RegisterGradient("Featurize")
# featurization takes in (x,y,z,a,offsets,input gradients for hcno) and returns a collected dx,dy,dz
def _feat_grad(op, grad_hs, grad_cs, grad_ns, grad_os):
    x,y,z,a,mo,macs,sis,acs = op.inputs
    fh, fc, fn, fo = op.outputs
    dLdx, dLdy, dLdz = ani_mod.featurize_grad(
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
        dLdx,
        dLdy,
        dLdz,
        None, # dLda
        None, # dLdmo
        None, # dLdmacs
        None, # dLdsis
        None, # dLdacs
    ]

# let g = a * b where:
# g = dE/dx
# a = dE/df
# b = df/dx
# the gradient of g is:
# dL/da = dL/dg * dg/da
# note that dg/da is simply b
# hence dL/da = dL/dg * b whereby the accumulation is now over
# the feature parameters (as opposed to the coordinates)
# ask yutong for how this works if you can't figure it out
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
        dLdz
    )

    return [
        None, # x
        None, # y
        None, # z
        None, # a
        None, # mo
        None, # macs
        None, # sis
        None, # acs
        dh, dc, dn, do
    ]

#(ytz: TODO) Add second derivative of this op to allow for training both charge and gradients
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
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    
    Parameters
    -----------
    tower_grads: List of lists of (gradient, variable) tuples. 
        The outer list is over individual gradients. The inner list is over the gradient
        calculation for each tower.

    Returns
    -------
    List of pairs of (gradient, variable)
        The gradient and its corresponding variable has been averaged across all towers.

    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        if len(grads) == 0:
            continue

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
        precision,
        layer_sizes=(128, 128, 64, 8, 1),
        activation_fn=activations.softplus_origin, #activations.celu,
        fit_charges=False):
        """
        A queue-enabled multi-gpu trainer. Construction of this class will also
        finalize and initialize all the variables pertaining to the input session.

        Parameters
        ----------
        sess: tf.Session
            A tensorflow session under which we use

        layer_Sizes: sequence of ints
            Defines the shapes of the intermediate atomic nn layers

        fit_charges: bool
            Whether or not we fit partial charges

        precision: tf.dtype
            Should be either tf.float32 or tf.float64

        """
        self.towers = towers
        self.num_towers = len(towers)

        assert fit_charges is False
        assert (precision is tf.float32) or (precision is tf.float64)
        assert self.num_towers > 0
        self.precision = precision

        self.x_enq = tf.placeholder(dtype=precision)
        self.y_enq = tf.placeholder(dtype=precision)
        self.z_enq = tf.placeholder(dtype=precision)
        self.a_enq = tf.placeholder(dtype=tf.int32)
        self.m_enq = tf.placeholder(dtype=tf.int32)
        self.yt_enq = tf.placeholder(dtype=precision)
        self.bi_enq = tf.placeholder(dtype=tf.int32)

        dtypes=[
            precision,  # Xs
            precision,  # Ys
            precision,  # Zs
            tf.int32,   # As
            tf.int32,   # mol ids
            precision,  # Y TRUEss
            tf.int32,   # b_idxs
        ]

        qtypes = [
            self.x_enq,
            self.y_enq,
            self.z_enq,
            self.a_enq,
            self.m_enq,
            self.yt_enq,
            self.bi_enq,
        ]

        # force fitting
        self.force_enq_x = tf.placeholder(dtype=precision, name="dx") # (batch_size, 1)
        self.force_enq_y = tf.placeholder(dtype=precision, name="dy") # (batch_size, 1)
        self.force_enq_z = tf.placeholder(dtype=precision, name="dz") # (batch_size, 1)
        dtypes.extend([precision, precision, precision])
        qtypes.extend([self.force_enq_x, self.force_enq_y, self.force_enq_z])

        queue = tf.FIFOQueue(capacity=20*self.num_towers, dtypes=dtypes);

        self.put_op = queue.enqueue(qtypes)
        self.sess = sess
        self.non_trainable_variables = []

        with tf.device('/cpu:0'):
            self.learning_rate = tf.get_variable('learning_rate', tuple(), precision, tf.constant_initializer(1e-3), trainable=False)
            self.optimizer = NadamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-8) # default is 1e-8

            self.global_step = tf.get_variable('global_step', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
            self.decr_learning_rate = tf.assign(self.learning_rate, tf.multiply(self.learning_rate, 0.7))
            self.global_epoch_count = tf.get_variable('global_epoch_count', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
            self.local_epoch_count = tf.get_variable('local_epoch_count', tuple(), tf.int32, tf.constant_initializer(0), trainable=False)
            self.incr_global_epoch_count = tf.assign(self.global_epoch_count, tf.add(self.global_epoch_count, 1))
            self.incr_local_epoch_count = tf.assign(self.local_epoch_count, tf.add(self.local_epoch_count, 1))
            self.reset_local_epoch_count = tf.assign(self.local_epoch_count, 0)

            # these data elements are unordered since the gpu grabs the batches in different orders
            self.tower_grads = [] # average is order invariant
            self.tower_force_grads = [] # yell at yutong for naming this
            self.tower_preds = []
            self.tower_bids = []
            self.tower_l2s = []
            self.tower_mos = []
            self.tower_coord_grads = []
            self.tower_features = []
            self.all_models = []
            self.tower_exp_loss = []
            self.tower_force_rmses = []
            self.parameters = []

            # parameters within a tower are shared.
            with tf.variable_scope(tf.get_variable_scope()):
                for tower_idx, tower_device in enumerate(towers):
                    with tf.device(tower_device):
                        with tf.name_scope("%s_%d" % ("tower", tower_idx)) as scope:

                            with tf.device('/cpu:0'):
                                get_op = queue.dequeue()
                                x_deq, y_deq, z_deq, a_deq, m_deq, labels, bi_deq = get_op[0], get_op[1], get_op[2], get_op[3], get_op[4], get_op[5], get_op[6]

                                dx_deq, dy_deq, dz_deq = get_op[7], get_op[8], get_op[9]

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

                            # print(f0.shape, f1.shape, f2.shape, f3.shape)

                            self.tower_features.append(tf.gather(
                                tf.concat([f0, f1, f2, f3], axis=0),
                                gather_idxs
                            ))

                            tower_model_near = MoleculeNN(
                                type_map=["H", "C", "N", "O"],
                                precision=precision,
                                atom_type_features=[f0, f1, f2, f3],
                                gather_idxs=gather_idxs,
                                layer_sizes=(feat_size,) + layer_sizes,
                                activation_fn=activation_fn,
                                prefix="near_")

                            # avoid duplicate parameters from later towers since the variables are shared.
                            if tower_idx == 0:
                                self.parameters.extend(tower_model_near.get_parameters())

                            self.all_models.append(tower_model_near)
                            tower_near_energy = tf.segment_sum(tower_model_near.atom_outputs, m_deq)

                            if fit_charges:
                                tower_model_charges = MoleculeNN(
                                    type_map=["H", "C", "N", "O"],
                                    atom_type_features=[f0, f1, f2, f3],
                                    gather_idxs=gather_idxs,
                                    layer_sizes=(feat_size,) + layer_sizes,
                                    precision=precision,
                                    prefix="charge_")

                                if tower_idx == 0:
                                    self.parameters.extend(tower_model_charges.get_parameters())

                                self.all_models.append(tower_model_charges)
                                tower_charges = tower_model_charges.atom_outputs

                                # (ytz + stevenso): we want to normalize the compute the charge per molecule
                                # note that this only works for *neutral* molecules. For molecules that have a formal charge
                                # we want to specify correct differently, or turn off the normalization entirely.
                                # tower_charges_per_mol = tf.segment_sum(tower_charges, m_deq) # per molecule charge
                                # tower_charges_per_mol = tf.divide(tower_charges_per_mol, tf.cast(mol_atom_counts, dtype=precision)) # per molecule avg charge
                                # tower_charge_correction = tf.gather(tower_charges_per_mol, m_deq) # generate the per atom correction
                                # tower_charges = tf.subtract(tower_charges, tower_charge_correction) # zero out the charge

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

                            tf.get_variable_scope().reuse_variables()

                            self.tower_preds.append(tower_pred)
                            tower_l2 = tf.squared_difference(tower_pred, labels)

                            self.tower_l2s.append(tower_l2)
                            tower_rmse = tf.sqrt(tf.reduce_mean(tower_l2))
                            tower_exp_loss = tf.exp(tf.cast(tower_rmse, dtype=tf.float64))
                            self.tower_exp_loss.append(tower_exp_loss)
                            tower_grad = self.optimizer.compute_gradients(tower_exp_loss)
                            self.tower_grads.append(tower_grad)

                            p_dx, p_dy, p_dz = tf.gradients(tower_pred, [x_deq, y_deq, z_deq])

                            self.tower_coord_grads.append([p_dx, p_dy, p_dz])

                            # forces are the negative of the gradient
                            f_dx, f_dy, f_dz = -p_dx, -p_dy, -p_dz

                            # optionally fit to the forces
                            dx_l2 = tf.pow(f_dx - dx_deq, 2)
                            dy_l2 = tf.pow(f_dy - dy_deq, 2)
                            dz_l2 = tf.pow(f_dz - dz_deq, 2)
                            dx_l2 = tf.sqrt(tf.reduce_mean(dx_l2))
                            dy_l2 = tf.sqrt(tf.reduce_mean(dy_l2))
                            dz_l2 = tf.sqrt(tf.reduce_mean(dz_l2))
                            # (todo): triple check that F = -grad(V)
                            tower_force_rmse = dx_l2 + dy_l2 + dz_l2
                            self.tower_force_rmses.append(tower_force_rmse)
                            tower_force_exp_loss = tf.exp(tf.cast(tower_force_rmse, dtype=tf.float64))

                            tower_force_grad = self.optimizer.compute_gradients(tower_force_exp_loss)
                            self.tower_force_grads.append(tower_force_grad)

            def tower_grads(grads):
                grads, vs = zip(*average_gradients(grads))
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                apply_gradient_op = self.optimizer.apply_gradients(zip(grads,vs), global_step=self.global_step)
                variable_averages = tf.train.ExponentialMovingAverage(0.9999, self.global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                return tf.group(apply_gradient_op, variables_averages_op)

            self.train_op = tower_grads(self.tower_grads)
            self.train_op_forces = tower_grads(self.tower_force_grads)

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

    def save_numpy(self, npz_file):
        """
        Save the parameters into a numpy npz. For the sake of consistency, we require that
        the npz_file ends in .npz

        .. note:: This saves the entire state of all variables (including non-trainable ones
            like the learning rate, etc.)

        Parameters
        ----------
        npz_file: str
            filename to save under. Must end in .npz

        """
        _, file_ext = os.path.splitext(npz_file)
        assert file_ext == ".npz"
        save_objs = {}
        all_vars = tf.global_variables()
        for var, val in zip(all_vars, self.sess.run(all_vars)):
            save_objs[var.name] = val
        np.savez(npz_file, **save_objs)

    def load_numpy(self, npz_file, strict=True):
        """
        Load a numpy checkpoint file.

        Parameters
        ----------
        npz_file: str
            filename to load

        strict: bool (optional)
            Whether or not we allow type conversions. By default
            this is set to True. If you're converting a 64 bit checkpoint file
            into lossy 32bit (and vice versa), you can set strict to False to enable the conversion
            automatically.

        """
        objs = np.load(npz_file, allow_pickle=False)
        assign_ops = []
        for k in objs.keys():
            tfo = self.sess.graph.get_tensor_by_name(k)
            npa = objs[k]
            if tfo.dtype.as_numpy_dtype != npa.dtype and strict is True:
                msg = "Cannot deserialize " + str(tfo.dtype.as_numpy_dtype) + " into " + str(npa.dtype)
                msg += ". You may want to set strict=False."
                raise TypeError(msg)
            assign_ops.append(tf.assign(tfo, objs[k].astype(tfo.dtype.as_numpy_dtype)))
        self.sess.run(assign_ops)

    def save(self, save_dir):
        """
        (DEPRECATED) Save the entire model to a given directory. Use save_numpy instead.

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
        (DEPRECATED) Load an existing model from an existing directory and initialize
        the trainer's Session variables. Use load_numpy instead.

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
        """
        Evaluates the absolute RMSE in kcal/mols of the y-values given dataset.

        Parameters
        ----------
        dataset: khan.RawDataset
            Dataset for evaluation

        batch_size: int (optional)
            Size of each batch used during prediction.

        Returns
        -------
        float
            A scalar for the RMSE of the dataset

        """
        # Todo: add support for force errors.
        test_l2s = self.feed_dataset(dataset, shuffle=False, target_ops=[self.unordered_l2s], batch_size=batch_size)
        return np.sqrt(np.mean(flatten_results(test_l2s))) * HARTREE_TO_KCAL_PER_MOL

    def coordinate_gradients(self, dataset, batch_size=1024):
        """
        Compute gradients with respect to the (x,y,z) coordinates of a dataset.

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
                # mo is an exclusive prefix sum so the first element is zero
                mo = mo[1:]
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

        .. note:: This should be used for investigative/debug purposes only. This returns tensors that are
            extremely large in size (hint: 600GB if iterating over gdb8 dataset)

        """

        results = self.feed_dataset(dataset, shuffle=False, target_ops=[self.tower_features, self.tower_mos, self.tower_bids], batch_size=batch_size)

        for (feats, mos, tids) in results:
            bidxs = np.argsort(tids)
            sorted_mos = np.take(mos, bidxs, axis=0)
            sorted_feats = np.take(feats, bidxs, axis=0)

            for (mo, feat) in zip(sorted_mos, sorted_feats):
                # mo is an exclusive prefix sum so the first element is zero
                mo = mo[1:]
                feats = np.split(feat, mo)
                for f in feats:
                    yield f


    def predict(self, dataset, batch_size=2048):
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

    def eval_rel_rmse(self, dataset, group_ys, batch_size=1024):
        """
        Evaluates the relative RMSE in kcal/mols of the y-values given dataset.

        Parameters
        ----------
        dataset: khan.RawDataset
            Dataset for evaluation. The y-values are ignored.

        group_ys: list of list of floats
            group_ys will be used in-place of the dataset's true y values.

        batch_size: int (optional)
            Size of each batch used during prediction.

        Returns
        -------
        float
            A scalar for the RMSE of the dataset

        """
        ordered_ys = self.predict(dataset, batch_size)
        return ed_harder_rmse(group_ys, ordered_ys) * HARTREE_TO_KCAL_PER_MOL

    def eval_eh_rmse(self, dataset, group_ys, batch_size=1024):
        """
        (DEPRECATED) renamed to eval_rel_rmse
        """
        return self.eval_rel_rmse(dataset, group_ys, batch_size)

    def feed_dataset(self,
        dataset,
        shuffle,
        target_ops,
        batch_size,
        fuzz=None,
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
                for b_idx, (mol_xs, mol_idxs, mol_yts, mol_grads) in enumerate(dataset.iterate(batch_size=batch_size, shuffle=shuffle, fuzz=fuzz)):
                    atom_types = (mol_xs[:, 0]).astype(np.int32)
                    if before_hooks:
                        self.sess.run(before_hooks)

                    feed_dict = {
                        self.x_enq: mol_xs[:, 1],
                        self.y_enq: mol_xs[:, 2],
                        self.z_enq: mol_xs[:, 3],
                        self.a_enq: atom_types,
                        self.m_enq: mol_idxs,
                        self.yt_enq: mol_yts,
                        self.bi_enq: b_idx
                    }

                    if mol_grads is not None:
                        feed_dict[self.force_enq_x] = mol_grads[:, 0]
                        feed_dict[self.force_enq_y] = mol_grads[:, 1]
                        feed_dict[self.force_enq_z] = mol_grads[:, 2]
                    else:
                        num_mols = mol_xs.shape[0]
                        feed_dict[self.force_enq_x] = np.zeros((num_mols, 0), dtype=self.precision.as_numpy_dtype)
                        feed_dict[self.force_enq_y] = np.zeros((num_mols, 0), dtype=self.precision.as_numpy_dtype)
                        feed_dict[self.force_enq_z] = np.zeros((num_mols, 0), dtype=self.precision.as_numpy_dtype)

                    self.sess.run(self.put_op, feed_dict=feed_dict)
                    g_b_idx += 1

                # division across multiple towers
                remainder = n_batches % self.num_towers
                if remainder:
                    for _ in range(self.num_towers - remainder):
                        if before_hooks:
                            self.sess.run(before_hooks)

                        feed_dict = {
                            self.x_enq: np.zeros((0, 1), dtype=self.precision.as_numpy_dtype),
                            self.y_enq: np.zeros((0, 1), dtype=self.precision.as_numpy_dtype),
                            self.z_enq: np.zeros((0, 1), dtype=self.precision.as_numpy_dtype),
                            self.a_enq: np.zeros((0,), dtype=np.int32),
                            self.m_enq: np.zeros((0,), dtype=np.int32),
                            self.yt_enq: np.zeros((0,)),
                            self.bi_enq: b_idx,
                        }

                        feed_dict[self.force_enq_x] = np.zeros((0, 1), dtype=self.precision.as_numpy_dtype)
                        feed_dict[self.force_enq_y] = np.zeros((0, 1), dtype=self.precision.as_numpy_dtype)
                        feed_dict[self.force_enq_z] = np.zeros((0, 1), dtype=self.precision.as_numpy_dtype)

                        self.sess.run(self.put_op, feed_dict=feed_dict)
                        b_idx += 1

            except Exception as e:
                print("QueueError:", e)

        executor = ThreadPoolExecutor(4)
        executor.submit(submitter)

        n_tower_batches = -(-dataset.num_batches(batch_size=batch_size) // self.num_towers)

        for i in range(n_tower_batches):
            yield self.sess.run(target_ops)

    # (ytz) - this is maintained by jminuse for the sake of convenience for now.
    # This is HOTMERGED - I'd avoid calling this code if possible, it seriously needs refactoring
    def train(self, save_dir, rd_train, rd_test, rd_gdb11, eval_names, eval_datasets, eval_groups, batch_size, max_local_epoch_count=25, max_batch_size=1e4, min_learning_rate=1e-7, max_global_epoch_count=1000):
        train_ops = [
            self.global_epoch_count,
            self.learning_rate,
            self.local_epoch_count,
            self.unordered_l2s,
            self.train_op
        ]
        start_time = time.time()
        best_test_score = self.eval_abs_rmse(rd_test)
        global_epoch = 0
        os.makedirs(save_dir)
        while batch_size < max_batch_size and self.sess.run(self.learning_rate)>min_learning_rate and global_epoch <= max_global_epoch_count: # bigger batches as fitting goes on, makes updates less exploratory
            while self.sess.run(self.local_epoch_count) < max_local_epoch_count and global_epoch <= max_global_epoch_count:
                for step in range(1): # how many rounds to perform before checking test rmse. Evaluation takes about as long as training for the same number of points, so it can be a waste to evaluate every time. 
                    train_step_time = time.time()
                    train_results = list( self.feed_dataset(
                        rd_train,
                        shuffle=True,
                        target_ops=train_ops,
                        batch_size=batch_size,
                        before_hooks=self.max_norm_ops,
                        fuzz=1e-4) )
                    train_abs_rmse = np.sqrt(np.mean(flatten_results(train_results, pos=3))) * HARTREE_TO_KCAL_PER_MOL
                    #print('%s Training step %d: train RMSE %.2f kcal/mol in %.1fs' % (save_dir, step, train_abs_rmse, time.time()-train_step_time) )
                global_epoch = train_results[0][0]
                learning_rate = train_results[0][1]
                local_epoch_count = train_results[0][2]
                test_abs_rmse_time = time.time()
                test_abs_rmse = self.eval_abs_rmse(rd_test)
                #print('test_abs_rmse_time', time.time()-test_abs_rmse_time )
                time_per_epoch = time.time() - start_time
                start_time = time.time()
                print(save_dir, end=' ')
                print(time.strftime("%Y-%m-%d %H:%M:%S"), 'tpe:', "{0:.2f}s,".format(time_per_epoch), 'g-epoch', global_epoch, 'l-epoch', local_epoch_count, 'lr', "{0:.0e}".format(learning_rate), 'batch_size', batch_size, '| train/test abs rmse:', "{0:.2f} kcal/mol,".format(train_abs_rmse), "{0:.2f} kcal/mol".format(test_abs_rmse), end='')

                if test_abs_rmse < best_test_score:
                    self.save_numpy(save_dir+'/best.npz')
                    best_test_score = test_abs_rmse
                    self.sess.run([self.incr_global_epoch_count, self.reset_local_epoch_count])
                else:
                    self.sess.run([self.incr_global_epoch_count, self.incr_local_epoch_count])

                gdb11_abs_rmse = self.eval_abs_rmse(rd_gdb11)
                print(' | gdb11 abs rmse', "{0:.2f} kcal/mol | ".format(gdb11_abs_rmse), end='')
                for name, ff_data, ff_groups in zip(eval_names, eval_datasets, eval_groups):
                    print(name, "abs/rel rmses", "{0:.2f} kcal/mol,".format(self.eval_abs_rmse(ff_data)), \
                        "{0:.2f} kcal/mol | ".format(self.eval_eh_rmse(ff_data, ff_groups)), end='')
                
                print('')
                self.save(save_dir)
                
            print("========== Decreasing learning rate, increasing batch size ==========")
            self.load_numpy(save_dir+'/best.npz')
            self.sess.run(self.decr_learning_rate)
            self.sess.run(self.reset_local_epoch_count)
            batch_size += 256
            #max_local_epoch_count = int(max_local_epoch_count*1.2) # increase this since higher batch size means fewer actual gradient steps per epoch

