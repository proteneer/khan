""" Implementation of a committee for training and inference """

import os
import numpy as np
import tensorflow as tf
from collections import namedtuple
from multiprocessing.dummy import Pool as ThreadPool

from khan.utils.constants import KCAL_MOL_IN_HARTREE
from khan.data.dataset import RawDataset
from khan.training.trainer_multi_tower import Trainer, flatten_results

import data_utils

NN_LAYERS = (256, 256, 64, 1)

CommitteeData = namedtuple(
    "CommitteeData",
    ["energy", "variance", "energy_grad", "variance_grad", "predictions"]
)


def model_E_and_grad(rd, scope_name, model, compute_gradient=False, add_atomic_energy=True):
    """
    compute energy and gradient over a dataset
        Parameters
        ---------- 
            X: definittion of molecule in khan format
            model: TrainerMultiTower instance
            compute_gradient: if True compute gradient else return zero vectors
            add_atomic_energy: if True, add in the sum of atomic energies
        Returns
        ---------
            list of energies and list of gradients
    """
    with tf.variable_scope(scope_name):
        energies = list(model.predict(rd))
        if compute_gradient:
            gradient = list(model.coordinate_gradients(rd))
        else:
            gradient = [np.zeros(len(mol)) for mol in rd.all_Xs]

    # add self energy correction
    if add_atomic_energy:
        for i, mol in enumerate(rd.all_Xs):
            atom_energies = [data_utils.selfIxnNrgWB97X[int(at[0])] for at in mol]
            energies[i] += sum(atom_energies) 

    return energies, gradient 

class Committee(object):

    MAX_NETWORKS = 100

    def __init__(
        self,
        save_dir,
        nmembers,
        tf_sess,
        precision,
        add_atomic_energy,
        ewc_read_name,
        **kwargs):
        """
        Start committee 

        Parameters
        ---------
            save_dir: a single directory that possible stores pre-trained networks
            nmembers: number of members desired (None will read all saved networks)
            tf_sess: TensorFlow session
            precision: precision for training/inference
            add_atomic_energy: indicates we want to add in atomic energies
            ewc_read_name: base name for EWC parameter files if EWC is desired
            kwargs: keyword arguments passed to Trainer
        """
        self._save_dir = save_dir
        self._sess = tf_sess
        self._add_atomic_energy = add_atomic_energy
        self._read_networks(nmembers, precision, ewc_read_name, **kwargs)
        self._training_data = [RawDataset([], []) for i in range(self._nmembers)]
        self._thread_pool = ThreadPool(self._nmembers)

    def _read_networks(self, nmembers, precision, ewc_read_name,  **kwargs):
        """
        read in saved networks if possible, else initialize
        if nmembers > the number of saved networks the remaining networks
        are initialized.

        Input:
            nmembers: desired number of networks, if None we read set the number of networks
                 from the number of found networks
            precision: precision for training/inference
            ewc_read_name: base name for ewc files, if None, we assume that we are not
                           using EWC
            kwargs: keyword arguments passed to Trainer
        """

        if not os.path.exists(self._save_dir):
            print("Save directory, %s, does not existing... creating" % self._save_dir)
            os.makedirs(self._save_dir)

        if nmembers is None:
            for i in range(self.MAX_NETWORKS):
                network_name = "committee-member-%d.npz" % i
                if not os.path.exists(os.path.join(self._save_dir, network_name)):
                    nmembers = i
                    break
            print("Found %d saved networks" % nmembers)

        if nmembers is None or nmembers < 1:
            raise IOError("Could not find enough saved networks")
            
        trainers = []
        for m in range(nmembers):
            scope_name = "committee-member-%d" % m
            save_file = os.path.join(self._save_dir, scope_name + ".npz")
            ewc_file = None 
            if ewc_read_name is not None:
                #untested
                ewc_file = self._ewc_filename(ewc_read_name, imember)

            with tf.variable_scope(scope_name):
                trainer = Trainer(self._sess, precision, ewc_read_file=ewc_file, **kwargs)
                print("initializing network %d" % m)
                trainer.initialize()

            if os.path.exists(save_file):
                print("loading network from saved file %s" % save_file)
                trainer.load_numpy(save_file, strict=False)

            trainers.append((scope_name, save_file, trainer))

        self._members = trainers
        self._nmembers = len(self._members) 


    def add_training_data(self, Xs, ys):
        """
        Add training data.  This data is split amongst the
        members, which each train on a separate training set

        Parameters
        ----------
        Xs: list of molecules (np arrays of shape natom x 4)
        ys: list of yvalues
        """
        assert len(Xs) == len(ys)

        # randomize ordering
        pairs = list(zip(Xs, ys))
        ndata = len(pairs)
        np.random.shuffle(pairs)

        # split up data evenly
        n_per_sample = ndata // self._nmembers
        for i in range(self._nmembers):
            istart = i * n_per_sample
            iend = (i + 1) * n_per_sample
            Xsample, ysample = zip(*pairs[istart:iend])

            self._training_data[i].all_Xs.extend(Xsample)
            self._training_data[i].all_ys.extend(ysample)

        # put each remaining datapoint into a randomly chosen network
        for idata in range(iend, ndata):
            imember = np.random.random_integers(0, self._nmembers)
            self._training_data[imember].all_Xs.append(pairs[idata][0])
            self._training_data[imember].all_ys.append(pairs[idata][1])
           

    def train(self, batch_size=256):
        """
        Run one epoch for each member.

        Returns
        --------
        global_epoch_count, local_epoch_count, learning_rate, mean error
        """

        train_args = [
            (i, rd_train, batch_size) for i, rd_train in enumerate(self._training_data)
        ]

        results = self._thread_pool.starmap(self._train_member, train_args)
        
        ave_train_results = [np.mean(lst) for lst in zip(*results)]

        return ave_train_results


    def _train_member(self, imember, rd_train, batch_size):
        """
        Run one epoch of training for the imember'th network

        Parameters:
        -----------
            imember: int
            rd_train: RawDataset
        Returns:
        -----------
            (global_epoch, local_epoch, lr, rmse training error)
        """

        assert imember >= 0 and imember < self._nmembers

        name, save_file, member = self._members[imember]

        train_ops = [
            member.global_epoch_count,
            member.learning_rate,
            member.local_epoch_count,
            member.l2s,
            member.train_op,
        ]

        with tf.variable_scope(name):
            train_results = list(member.feed_dataset(
                rd_train,
                shuffle=True,
                target_ops=train_ops,
                batch_size=batch_size,
            ))

        global_epoch = train_results[0][0]
        learning_rate = train_results[0][1]
        local_epoch = train_results[0][2]
        train_abs_rmse = np.sqrt(
            np.mean(flatten_results(train_results, pos=3))
        ) * KCAL_MOL_IN_HARTREE 

        return global_epoch, local_epoch, learning_rate, train_abs_rmse

    def run_session_ops(self, op_names):
        """
        run a list of ops with a tf Session

        Inputs:
        -------
            sess: tf.Session
            op_names: list of strings (ops)
        Returns:
        -------
        list of results from running the ops
        the length of the list is equal to the number
        of members 
        """
        results = []
        for name, save_file, member in self._members:
            member_ops = [
                getattr(member, op) for op in op_names
            ]
            with tf.variable_scope(name):
                results.append(
                    self._sess.run(member_ops)
                )

        return results

    def _run_member_method(self, imember, func_name, *args, **kwargs):
        """
        run a member function of a committee member

        Inputs:
        -------
            imember: member number
            func_name: name of function to run
            args: positional args passed to function
            kwargs: keyword args passed to function
        Returns:
        --------
        return value of function
        """

        # should be threaded I suppose
        assert imember >= 0 and imember < self._nmembers

        name, save_file, member = self._members[imember]
        with tf.variable_scope(name):
            func = getattr(member, func_name)
            result = func(*args, **kwargs)

        return result

    def eval_abs_rmse(self, rd):
        """
        Evaluate the committee rmse over the dataset

        Parameters:
        -----------
            rd: RawDataset
        Returns:
        -----------
            rmse error in kcal/mol
        """
        pairs = self.energy_and_variance(rd)
        e, s2 = map(np.array, zip(*pairs))
        errors = e - np.array(rd.all_ys)

        rmse = np.sqrt(np.mean(errors[:]**2.0)) * KCAL_MOL_IN_HARTREE

        return rmse

    def energy_and_variance(self, rd):
        """
        return predicted energy and variance for a dataset

        Parameters
        -----------
            rd: RawDataset
        Returns:
        ----------
            a list of energy, variance pairs
        """
        
        return [(instance.energy, instance.variance) for instance in self.compute_data(rd)]


    def compute_data(self, rd):
        """
        Computes data over a dataset

        Parameters
        -----------
            rd: RawDataset
        Returns:
        ----------
            a list of ComitteeData instances
        """
        n_data = rd.num_mols()

        # evaluate energy/gradient data for all members for all data
        energy_args = []
        for imember, (scope_name, save_file, member) in enumerate(self._members):
            energy_args.append(
                (rd, scope_name, member, False, self._add_atomic_energy)
            )

        results = self._thread_pool.starmap(model_E_and_grad, energy_args)

        energies, gradients = [], []
        for e, g in  results:
            energies.append(e)
            gradients.append(g)

        # accumulate ave and variance of energies across members of committee
        # do this for every data point
        comittee_data = []
        for idata in range(n_data):

            energy_ave = 0.0
            energy_grad = 0.0
            for imember in range(self._nmembers):
                energy_ave += energies[imember][idata]
                energy_grad += gradients[imember][idata]
            energy_ave /= self._nmembers
            energy_grad /= self._nmembers

            variance = 0.0
            variance_grad = 0.0
            for imember in range(self._nmembers):
                variance += (energies[imember][idata] - energy_ave)**2.0
                variance_grad += 2.0 * (energies[imember][idata] - energy_ave) \
                                     * (gradients[imember][idata] - energy_grad)
            variance /= self._nmembers
            variance_grad /= self._nmembers

            dumb = [energies[imember][idata] for imember in range(self._nmembers)]
            data = CommitteeData(energy_ave, variance, energy_grad, variance_grad, dumb)
            comittee_data.append(data)

        return comittee_data

    def _ewc_filename(self,  base, imember):
        """
        Generate a filename for the imemberth members ewc parameters
        """
        assert not base.endswith(".npz")
        ewc_save_file = os.path.join(
            self._save_dir,
            base + "-member-" + str(imember) + ".npz"
        )
        return ewc_save_file 

    def save_ewc_params(self, ewc_base_name):
        """
        Save EWC parameters to a file
        the ewc_save_file is mangled to result in a file
        for each network.

        Parameters
        ---------
        ewc_base_name: base name of file to save ewc parameters to
        """

        base = ewc_base_name
        if ewc_base_name.endswith(".npz"):
            base = os.path.splitext(ewc_base_name)[0]
        
        ewc_args = []
        for i, rd in enumerate(self._training_data):
            ewc_save_file = self._ewc_filename(base, i)
            ewc_args.append(
                (i, "save_ewc_params", rd, ewc_save_file)
            )

        self._thread_pool.starmap(self._run_member_method, ewc_args) 

    def save(self):
        """
        Save the networks
        """
        # save/read is not quite right
        # trainer saves everything in tf.global_variables and you get a ton
        # of lookup failures when reading.  I think this is just because 
        # everything is saved in a single npz file yet we can only set
        # a certain members variables when reading. 

        print("Saving networks...")
        save_args = [
            (i, "save_numpy", save_file) for i, (_, save_file, _) in enumerate(self._members)
        ]
        
        #self._thread_pool.starmap(self._run_member_method, save_args)
        for args in save_args:
            self._run_member_method(*args)

