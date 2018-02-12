import glob
import os
import numpy as np
import tempfile
import time
import tensorflow as tf
import sklearn
import sklearn.model_selection

from khan.utils.helpers import ed_harder_rmse
from khan.model.nn import MoleculeNN, mnn_staging
from khan.training.trainer import Trainer
from khan.data import pyanitools as pya
from khan.data.dataset import RawDataset, FeaturizedDataset

from concurrent.futures import ThreadPoolExecutor

import argparse


HARTREE_TO_KCAL_PER_MOL = 627.509

selfIxnNrgWB97 = np.array([
    -0.500607632585,
    -37.8302333826,
    -54.5680045287,
    -75.0362229210], dtype=np.float32)

selfIxnNrgMO62x = np.array([
   -0.498135,
   -37.841399,
   -54.586413,
   -75.062826,
], dtype=np.float32)

import correction

def convert_species_to_atomic_nums(s):
  PERIODIC_TABLE = {"H": 0, "C": 1, "N": 2, "O": 3}
  res = []
  for k in s:
    res.append(PERIODIC_TABLE[k])
  return np.array(res, dtype=np.int32)


def load_calibration_file(calibration_file):
    with open(calibration_file, 'r') as fh:

        mapping = {}

        for line in fh.readlines():
            parts = line.split()
            path = parts[0].split(".")[0]
            energy = parts[-1]
            mapping[path] = float(energy)

        return mapping

def parse_xyz(xyz_file, use_fitted):
    """
    If use_fitted is False, return the mo62x atomization energies. Otherwise, return a fitted energy.
    """
    with open(xyz_file, "r") as fh:

        header = fh.readline()
        comment = fh.readline()

        y = float(comment.split()[-1])
        body = fh.readlines()
        elems = []
        coords = []

        for line in body:
            res = line.split()
            elem = res[0]
            elems.append(elem)
            coords.append((float(res[1]),float(res[2]),float(res[3])))

        coords = np.array(coords, dtype=np.float32)

        Z = convert_species_to_atomic_nums(elems)

        mo62xoffset = 0
        js18pairwiseOffset = correction.jamesPairwiseCorrection_C(coords, Z)/HARTREE_TO_KCAL_PER_MOL

        for z in Z:
            mo62xoffset += selfIxnNrgMO62x[z]

        if use_fitted:
            y -= js18pairwiseOffset
        else:
            y -= mo62xoffset 

        R = np.array(coords, dtype=np.float32)
        X = np.concatenate([np.expand_dims(Z, 1), R], axis=1)

        return X, y


def load_ff_files_groups(ff_dir, use_fitted=False):
    ys = []

    for gidx, (root, dirs, files) in enumerate(os.walk(ff_dir)):
        group_ys = []
        for filename in files:
            rootname, extension = os.path.splitext(filename)
            if extension == ".xyz":
                filepath = os.path.join(root, filename)
                _, y = parse_xyz(filepath, use_fitted)
                group_ys.append(y)
            else:
                print("Unknown filetype:", filename)
        if len(group_ys) > 0:
            ys.append(group_ys)

    return ys

def load_ff_files(ff_dir, use_fitted=False):
    Xs = []
    ys = []
    for root, dirs, files in os.walk(ff_dir):
        for filename in files:
            rootname, extension = os.path.splitext(filename)
            if extension == ".xyz":
                filepath = os.path.join(root, filename)
                X, y = parse_xyz(filepath, use_fitted)
                Xs.append(X)
                ys.append(y)
            else:
                print("Unknown filetype:", filename)

    # import matplotlib.mlab as mlab
    # import matplotlib.pyplot as plt

    # n, bins, patches = plt.hist(ys, 300, facecolor='green', alpha=0.75)
    # plt.show()

    # assert 0
    # print(len(Xs), len(ys))
    return Xs, ys


def load_hdf5_files(
    hdf5files,
    calibration_map=None,
    energy_cutoff=100.0/HARTREE_TO_KCAL_PER_MOL,
    use_fitted=False):
    """
    Load the ANI dataset.

    Parameters
    ----------
    hdf5files: list of str
        List of paths to hdf5 files that will be used to generate the dataset. The data should be
        in the format used by the ANI-1 dataset.

    use_fitted: bool
        If use_fitted is False, return the mo62x atomization energies. Otherwise, return a fitted energy.


    Returns
    -------
    Dataset, list of int
        Returns a Dataset object and a list of integers corresponding to the groupings of the
        respective atoms.

    """

    Xs = []
    ys = []

    print("Loading...")

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

            path = P.split("/")[-1]

            Z = convert_species_to_atomic_nums(S)
            minimum_wb97 = np.amin(E)

            if use_fitted:

                calibration_offset = 0

                if calibration_map:
                    calibration_offset = calibration_map[path] - minimum_wb97 

                for k in range(len(E)):
                    if energy_cutoff is not None and E[k] - minimum_wb97 > energy_cutoff:
                        continue
                    js18pairwiseOffset = correction.jamesPairwiseCorrection_C(R[k], Z)/HARTREE_TO_KCAL_PER_MOL
                    y = E[k] - js18pairwiseOffset + calibration_offset
                    ys.append(y)
                    X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                    Xs.append(X)

            else:
                wb97offset = 0
                mo62xoffset = 0

                for z in Z:
                    wb97offset += selfIxnNrgWB97[z]
                    mo62xoffset += selfIxnNrgMO62x[z]

                calibration_offset = 0

                if calibration_map:
                    min_atomization_wb97 = minimum_wb97 - wb97offset
                    min_atomization_mo62x = calibration_map[path] - mo62xoffset
                    # difference between the wb97_min and the mo62x_min
                    calibration_offset = min_atomization_mo62x - min_atomization_wb97

                for k in range(len(E)):
                    if energy_cutoff is not None and E[k] - minimum_wb97 > energy_cutoff:
                        continue
                    y = E[k] - wb97offset + calibration_offset

                    ys.append(y)
                    X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                    Xs.append(X)


    # import matplotlib.mlab as mlab
    # import matplotlib.pyplot as plt

    # n, bins, patches = plt.hist(ys, 400, facecolor='green', alpha=0.75)
    # plt.show()

    # assert 0

    return Xs, ys


def flatten_results(res):
    flattened = []
    for l in res:
        flattened.append(l[0])
    return np.concatenate(flattened).reshape((-1,))



if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="Run ANI1 neural net training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fitted', default=False, action='store_true', help="Whether or use fitted or self-ixn")
    parser.add_argument('--add_ffdata', default=False, action='store_true', help="Whether or not to add the forcefield data")
    parser.add_argument('--prod', default=False, action='store_true', help="Whether we run over all of gdb8")

    parser.add_argument('--work-dir', default='~/work', help="location where work data is dumped")
    parser.add_argument('--train-dir', default='~/ANI-1_release', help="location where work data is dumped")

    args = parser.parse_args()
    print("Arguments", args)


    ANI_TRAIN_DIR = args.train_dir
    ANI_WORK_DIR = args.work_dir

    CALIBRATION_FILE_TRAIN = os.path.join(ANI_TRAIN_DIR, "results_QM_M06-2X.txt")
    CALIBRATION_FILE_TEST = os.path.join(ANI_TRAIN_DIR, "gdb_11_cal.txt")
    ROTAMER_TRAIN_DIR = os.path.join(ANI_TRAIN_DIR, "rotamers/train")
    ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "rotamers/test")


    # print("ANI TRAIN AND WORK DIRS", ANI_TRAIN_DIR, ANI_WORK_DIR)

    batch_size = 1024

    save_dir = os.path.join(ANI_WORK_DIR, "save")
    data_dir_train = os.path.join(ANI_WORK_DIR, "train")
    data_dir_test = os.path.join(ANI_WORK_DIR, "test")
    data_dir_gdb11 = os.path.join(ANI_WORK_DIR, "gdb11")
    data_dir_fftest = os.path.join(ANI_WORK_DIR, "fftest")

    use_fitted = args.fitted
    add_ffdata = args.add_ffdata

    print("use_fitted, add_ffdata", use_fitted, add_ffdata)

    if not os.path.exists(data_dir_train):

        cal_map_train = load_calibration_file(CALIBRATION_FILE_TRAIN)
        cal_map_test = load_calibration_file(CALIBRATION_FILE_TEST)


        if args.prod:
            gdb_files = [
                os.path.join(ANI_TRAIN_DIR, "ani_gdb_s01.h5"),
                os.path.join(ANI_TRAIN_DIR, "ani_gdb_s02.h5"),
                os.path.join(ANI_TRAIN_DIR, "ani_gdb_s03.h5"),
                os.path.join(ANI_TRAIN_DIR, "ani_gdb_s04.h5"),
                os.path.join(ANI_TRAIN_DIR, "ani_gdb_s05.h5"),
                os.path.join(ANI_TRAIN_DIR, "ani_gdb_s06.h5"),
                os.path.join(ANI_TRAIN_DIR, "ani_gdb_s07.h5"),
                os.path.join(ANI_TRAIN_DIR, "ani_gdb_s08.h5"),
            ]
        else:
            gdb_files = [os.path.join(ANI_TRAIN_DIR, "ani_gdb_s01.h5")]

        Xs, ys = load_hdf5_files(gdb_files, calibration_map=cal_map_train, use_fitted=use_fitted)


        print("Loading ff testing data...")
        ff_test_Xs, ff_test_ys = load_ff_files(ROTAMER_TEST_DIR, use_fitted=use_fitted)

        if add_ffdata:
            print("Loading ff training data...")
            ff_train_Xs, ff_train_ys = load_ff_files(ROTAMER_TRAIN_DIR, use_fitted=use_fitted)

            Xs.extend(ff_train_Xs) # add training data here
            ys.extend(ff_train_ys)

        # shuffle dataset
        Xs, ys = sklearn.utils.shuffle(Xs, ys)

        subsample_size = len(Xs)

        assert subsample_size <= len(Xs)

        Xs, ys = Xs[:subsample_size], ys[:subsample_size]

        print("lengths", len(Xs), len(ys))

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xs, ys, test_size=0.25)

        rd_train = RawDataset(X_train, y_train)
        rd_test  = RawDataset(X_test,  y_test)

        print("Loading gdb11 test set...")
        X_gdb11, y_gdb11 = load_hdf5_files([
            os.path.join(ANI_TRAIN_DIR, "ani1_gdb10_ts.h5"),
        ], calibration_map=cal_map_test, use_fitted=use_fitted)

        rd_gdb11  = RawDataset(X_gdb11, y_gdb11)
        rd_fftest  = RawDataset(ff_test_Xs, ff_test_ys)

        try:
            os.makedirs(data_dir_train, exist_ok=True)
            os.makedirs(data_dir_test, exist_ok=True)
            os.makedirs(data_dir_gdb11, exist_ok=True)
            os.makedirs(data_dir_fftest, exist_ok=True)
        except Exception as e:
            print("warning:", e)

        # print("featurizing...")
        fd_train = rd_train.featurize(batch_size, data_dir_train)
        fd_test = rd_test.featurize(batch_size, data_dir_test)
        fd_gdb11 = rd_gdb11.featurize(batch_size, data_dir_gdb11)
        fd_fftest = rd_fftest.featurize(batch_size, data_dir_fftest)

    else:

        fd_train = FeaturizedDataset(data_dir_train)
        fd_test = FeaturizedDataset(data_dir_test)
        fd_gdb11 = FeaturizedDataset(data_dir_gdb11)
        fd_fftest = FeaturizedDataset(data_dir_fftest)


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

    trainer = Trainer(
        mnn,
        yt_deq,
        f0_enq,
        f1_enq,
        f2_enq,
        f3_enq,
        gi_enq,
        mi_enq,
        yt_enq,
        put_op
    )
    # train_op_rmse = trainer.get_train_op_rmse()
    train_op_exp = trainer.get_train_op_exp()
    loss_op = trainer.get_loss_op()
    predict_op = trainer.model.predict_op()
    max_norm_ops = trainer.get_maxnorm_ops()

    sess = tf.Session()
    # saver = tf.train.Saver(tf.trainable_variables())
    saver = tf.train.Saver()


    save_prefix = "ani"
    save_path = os.path.join(save_dir, save_prefix)
    if os.path.exists(save_dir):

        checkpoints = glob.glob(save_path+"*.index")
        max_gstep = 0
        for f in checkpoints:
            g_step = int(f.split('.')[0].split("-")[1])
            if g_step > max_gstep:
                max_gstep = g_step

        last_file = save_path+"-"+str(max_gstep)
        print("Loading from epoch:", max_gstep // fd_train.num_batches(), "file:", last_file)

        saver.restore(sess, last_file)

    else:
        sess.run(tf.global_variables_initializer())
        # print("Pre-minimizing one epoch with rmse loss...")
    num_epochs = 2
    tot_time = 0
    
    print("num batches", fd_train.num_batches())

    st = time.time()
 
    l2_losses = [trainer.l2]

    # test_l2s = trainer.feed_dataset(sess, fd_test, shuffle=False, target_ops=l2_losses) 
    # best_test_score = np.sqrt(np.mean(np.concatenate(test_l2s).reshape((-1,))))

    max_local_epoch_count = 100

    train_ops = [trainer.global_step, trainer.learning_rate, trainer.rmse, train_op_exp]

    print("Loading ff testing group data...")
    ff_test_group_ys = load_ff_files_groups(ROTAMER_TEST_DIR, use_fitted=use_fitted)

    fftest_ys = trainer.feed_dataset(sess, fd_fftest, shuffle=False, target_ops=[trainer.model.predict_op()])
    fftest_eh_rmse = ed_harder_rmse(ff_test_group_ys, flatten_results(fftest_ys))
    fftest_l2s = trainer.feed_dataset(sess, fd_fftest, shuffle=False, target_ops=l2_losses)
    fftest_rmse = np.sqrt(np.mean(flatten_results(fftest_l2s)))

    print("fftest eh_rmse, rmse", fftest_eh_rmse, fftest_rmse)

    for lr in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        local_epoch_count = 0 # epochs within a learning rate

        print("setting learning rate to", lr)
        sess.run(tf.assign(trainer.learning_rate, lr))
        while local_epoch_count < max_local_epoch_count:

            # sess.run(max_norm_ops) # extraneous, do once at the beginning
            train_results = trainer.feed_dataset(sess, fd_train, shuffle=True, target_ops=train_ops)
            sess.run(max_norm_ops)
            global_epoch = train_results[0][0] // fd_train.num_batches()
            # global_epoch = 0

            test_l2s = trainer.feed_dataset(sess, fd_test, shuffle=False, target_ops=l2_losses)

            test_rmse = np.sqrt(np.mean(flatten_results(test_l2s)))

            if test_rmse < best_test_score:
                sp = saver.save(sess, save_path, global_step=trainer.global_step)

                gdb11_l2s = trainer.feed_dataset(sess, fd_gdb11, shuffle=False, target_ops=l2_losses)
                gdb11_rmse = np.sqrt(np.mean(flatten_results(gdb11_l2s)))

                fftest_ys = trainer.feed_dataset(sess, fd_fftest, shuffle=False, target_ops=[trainer.model.predict_op()])
                fftest_eh_rmse = ed_harder_rmse(ff_test_group_ys, flatten_results(fftest_ys))

                fftest_l2s = trainer.feed_dataset(sess, fd_fftest, shuffle=False, target_ops=l2_losses)
                fftest_rmse = np.sqrt(np.mean(flatten_results(fftest_l2s)))

                print("Better g-epo:", global_epoch, "| lr:", lr, \
                      "| l-epo:", local_epoch_count, \
                      "| test rmse:", test_rmse, "| gdb11 rmse:", gdb11_rmse, "| fftest rmse:", fftest_rmse, "| eh rmse:", fftest_eh_rmse)

                local_epoch_count = 0
                best_test_score = test_rmse
            else:

                print("Worse g-epo:", global_epoch, "| lr:", lr, \
                      "| l-epo:", local_epoch_count, \
                      "| test rmse:", test_rmse)
                local_epoch_count += 1


    tot_time = time.time() - st # this logic is a little messed up

    tpm = tot_time/(fd_train.num_batches()*batch_size*num_epochs*3)
    print("Time Per Mol:", tpm, "seconds")
    print("Samples per minute:", 60/tpm)
