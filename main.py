import os
import numpy as np
import tempfile
import time
import tensorflow as tf
import sklearn
import sklearn.model_selection

from khan.model.nn import MoleculeNN, mnn_staging
from khan.training.trainer import Trainer
from khan.data import pyanitools as pya
from khan.data.dataset import RawDataset, FeaturizedDataset

from concurrent.futures import ThreadPoolExecutor

HARTREE_TO_KCAL_PER_MOL = 627.509

def convert_species_to_atomic_nums(s):
  PERIODIC_TABLE = {"H": 0, "C": 1, "N": 2, "O": 3}
  res = []
  for k in s:
    res.append(PERIODIC_TABLE[k])
  return np.array(res, dtype=np.int32)

def load_hdf5_files(hdf5files, energy_cutoff=100.0/HARTREE_TO_KCAL_PER_MOL):
    """
    Load the ANI dataset.

    Parameters
    ----------
    hdf5files: list of str
        List of paths to hdf5 files that will be used to generate the dataset. The data should be
        in the format used by the ANI-1 dataset.

    batch_size: int
        Used to determined the shard_size, where shard_size is batch_size * 4096

    data_dir: str
        Directory in which we save the resulting data

    mode: str
        Accepted modes are "relative", "atomization", or "absolute". These settings are used
        to adjust the dynamic range of the model, with absolute having the greatest and relative
        having the lowest. Note that for atomization we approximate the single atom energy
        using a different level of theory

    max_atoms: int
        Total number of atoms we allow for.

    energy_cutoff: int or None
        A cutoff to use for pruning high energy conformations from a dataset. Units are in
        hartrees. Default is set to 100 kcal/mol or ~0.16 hartrees.

    selection_size: int or None
        Subsample of conformations that we want to choose from gdb-8

    Returns
    -------
    Dataset, list of int
        Returns a Dataset object and a list of integers corresponding to the groupings of the
        respective atoms.

    """

    atomizationEnergies = np.array([
        0,
        -0.500607632585,
        -37.8302333826,
        -54.5680045287,
        -75.0362229210], dtype=np.float32)

    Xs = []
    ys = []

    for hdf5file in hdf5files:
        adl = pya.anidataloader(hdf5file)
        for data in adl:

            # Extract the data
            P = data['path']
            R = data['coordinates']
            E = data['energies']
            S = data['species']
            smi = data['smiles']

            print("Processing: ", P)

            Z = convert_species_to_atomic_nums(S)
            minimum = np.amin(E)

            offset = 0

            for z in Z:
                offset += atomizationEnergies[z]


            for k in range(len(E)):
                if energy_cutoff is not None and E[k] - minimum > energy_cutoff:
                    continue

                y = E[k] - offset

                # print(E[k], offset, y)
                ys.append(y)

                # print(np.expand_dims(Z, 1), R[k])

                X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                Xs.append(X)

            # debug
            if len(Xs) > 16000:
                break

    return Xs, ys

if __name__ == "__main__":

    data_dir_train = "/media/yutong/fast_datablob/v3/train"
    data_dir_test = "/media/yutong/fast_datablob/v3/train"


    batch_size = 1024


    if not os.path.exists(data_dir_train):

        Xs, ys = load_hdf5_files([
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s01.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s02.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s03.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s04.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s05.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s06.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s07.h5",
            "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s08.h5",
        ])
        # assert len(Xs) == len(ys)
        # shuffle dataset
        # perm = np.random.permutation(len(Xs))
        Xs, ys = sklearn.utils.shuffle(Xs, ys)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xs, ys, test_size=0.25)

        rd_train = RawDataset(X_train, y_train)
        rd_test  = RawDataset(X_test,  y_test)

        try:
            os.makedirs(data_dir_train, exist_ok=True)
            os.makedirs(data_dir_test, exist_ok=True)
        except Exception as e:
            print("warning:", e)

        fd_train = rd_train.featurize(batch_size, data_dir_train)
        fd_test = rd_test.featurize(batch_size, data_dir_test)

    else:

        fd_train = FeaturizedDataset(data_dir_train)
        fd_test = FeaturizedDataset(data_dir_test)


    # print("featurizing...")
    # fd = rd_train.featurize(batch_size=batch_size, data_dir=data_dir)
    


    (f0_enq, f1_enq, f2_enq, f3_enq, gi_enq, mi_enq, yt_enq), \
    (f0_deq, f1_deq, f2_deq, f3_deq, gi_deq, mi_deq, yt_deq), \
    put_op = mnn_staging()


    mnn = MoleculeNN(
        type_map=["H", "C", "N", "O"],
        atom_type_features=[f0_deq, f1_deq, f2_deq, f3_deq],
        gather_idxs=gi_deq,
        mol_idxs=mi_deq,
        layer_sizes=(384, 256, 128, 64, 1))

    trainer = Trainer(mnn, yt_deq)
    train_op_rmse = trainer.get_train_op_rmse()
    train_op_exp = trainer.get_train_op_exp()
    loss_op = trainer.get_loss_op()
    predict_op = trainer.model.predict_op()
    max_norm = trainer.get_maxnorm_ops()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_epochs = 2048

    def submitter():
        for _ in range(num_epochs):
            for b_idx, (f0, f1, f2, f3, gi, mi, yt) in enumerate(fd_train.iterate(shuffle=True)):

                # print(np.max(f0), f0)
                # print(np.max(f1), f1)
                # print(np.max(f2), f2)
                # print(np.max(f3), f3)

                # assert 0

                try:
                    sess.run(put_op, feed_dict={
                        f0_enq: f0,
                        f1_enq: f1,
                        f2_enq: f2,
                        f3_enq: f3,
                        gi_enq: gi,
                        mi_enq: mi,
                        yt_enq: yt,
                    })
                except Exception as e:
                    print("OMG WTF BBQ", e)

    executor = ThreadPoolExecutor(4)

    executor.submit(submitter)

    tot_time = 0
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    print("num batches", fd_train.num_batches())

    st = time.time()
    for e in range(num_epochs):
        print("epoch:", e)
        for i in range(fd_train.num_batches()):
            # print("running", i)
            # res = sess.run([train_op, loss_op])
            # res = sess.run([predict_op, train_op, loss_op])
            if e < 32:
                res = sess.run([max_norm, trainer.rmse, train_op_rmse])
            else:
                res = sess.run([max_norm, trainer.exp_loss, train_op_exp])
            print(res[1])

    tot_time = time.time() - st # this logic is a little messed up

    tpm = tot_time/(fd_train.num_batches()*batch_size*num_epochs)
    print("Time Per Mol:", tpm, "seconds")
    print("Samples per minute:", 60/tpm)