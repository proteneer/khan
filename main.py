import glob
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

# atomizationEnergiesWB97 = np.array([
#     -0.500607632585,
#     -37.8302333826,
#     -54.5680045287,
#     -75.0362229210], dtype=np.float32)

# atomizationEnergiesMO62x = np.array([
#    -0.498135,
#    -37.841399,
#    -54.586413,
#    -75.062826,
# ], dtype=np.float32)

# atomizationEnergiesJS18 = np.array([
#     -379.47160374/HARTREE_TO_KCAL_PER_MOL,
#     -23887.88837703/HARTREE_TO_KCAL_PER_MOL,
#     -34328.09841337/HARTREE_TO_KCAL_PER_MOL,
#     -47175.24402322/HARTREE_TO_KCAL_PER_MOL,
# ])

import correction

CALIBRATION_FILE = "/home/yutong/roitberg_data/ANI-1_release/results_QM_M06-2X.txt"

ROITBERG_ANI_DIR = "/home/yutong/roitberg_data/ANI-1_release"
ROTAMER_TRAIN_DIR = "/home/yutong/roitberg_data/ANI-1_release/rotamers/train"
ROTAMER_TEST_DIR = "/home/yutong/roitberg_data/ANI-1_release/rotamers/test"

WORK_DIR = "/media/yutong/nvme_ssd/v3/" # put this on a fast SSD with 1 TB of data

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

def parse_xyz(xyz_file):
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

        wb97offset = 0
        mo62xoffset = 0
        js18offset = 0
        js18pairwiseOffset = correction.jamesPairwiseCorrection_C(coords, Z)/HARTREE_TO_KCAL_PER_MOL

        for z in Z:
            wb97offset += atomizationEnergiesWB97[z]
            mo62xoffset += atomizationEnergiesMO62x[z]
            js18offset += atomizationEnergiesJS18[z]

        y = y - js18pairwiseOffset

        R = np.array(coords, dtype=np.float32)
        X = np.concatenate([np.expand_dims(Z, 1), R], axis=1)

        return X, y

def load_ff_files(ff_dir):
    Xs = []
    ys = []
    for root, dirs, files in os.walk(ff_dir):
        for filename in files:
            rootname, extension = os.path.splitext(filename)
            if extension == ".xyz":
                filepath = os.path.join(root, filename)
                X, y = parse_xyz(filepath)
                Xs.append(X)
                ys.append(y)
            else:
                print("Unknown filetype:", filename)

    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    n, bins, patches = plt.hist(ys, 100, facecolor='green', alpha=0.75)
    plt.show()

    assert 0
    print(len(Xs), len(ys))
    return Xs, ys


def load_hdf5_files(
    hdf5files,
    calibration_map=None,
    energy_cutoff=100.0/HARTREE_TO_KCAL_PER_MOL):
    """
    Load the ANI dataset.

    Parameters
    ----------
    hdf5files: list of str
        List of paths to hdf5 files that will be used to generate the dataset. The data should be
        in the format used by the ANI-1 dataset.

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
            minimum = np.amin(E)

            wb97offset = 0
            mo62xoffset = 0
            js18offset = 0 

            for z in Z:
                wb97offset += atomizationEnergiesWB97[z]
                mo62xoffset += atomizationEnergiesMO62x[z]
                js18offset += atomizationEnergiesJS18[z]

            calibration_offset = 0

            # temporarily disable
            # if calibration_map:
            #     min_atomization_wb97 = minimum - wb97offset
            #     min_atomization_mo62x = calibration_map[path] - mo62xoffset
            #     calibration_offset = min_atomization_mo62x - min_atomization_wb97

            if calibration_map:
                calibration_offset = calibration_map[path] - minimum 

            for k in range(len(E)):
                if energy_cutoff is not None and E[k] - minimum > energy_cutoff:
                    continue

                js18pairwiseOffset = correction.jamesPairwiseCorrection_C(R[k], Z)/HARTREE_TO_KCAL_PER_MOL
                y = E[k] - js18pairwiseOffset + calibration_offset
                # y = E[k] - js18pairwiseOffset

                # print(y)

                # y = E[k] - wb97offset + calibration_offset
                ys.append(y)
                X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                Xs.append(X)

    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    n, bins, patches = plt.hist(ys, 100, facecolor='green', alpha=0.75)
    plt.show()

    assert 0

    return Xs, ys




if __name__ == "__main__":

    batch_size = 1024

    save_dir = os.path.join(WORK_DIR, "save")
    data_dir_train = os.path.join(WORK_DIR, "train")
    data_dir_test = os.path.join(WORK_DIR, "test")
    data_dir_gdb11 = os.path.join(WORK_DIR, "gdb11")
    data_dir_fftest = os.path.join(WORK_DIR, "fftest")

    if not os.path.exists(data_dir_train):

        cal_map = load_calibration_file(CALIBRATION_FILE)

        Xs, ys = load_hdf5_files([
            os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s01.h5"),
            os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s02.h5"),
            os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s03.h5"),
            os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s04.h5"),
            os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s05.h5"),
            os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s06.h5"),
            os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s07.h5"),
            # os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s08.h5"),
        ], calibration_map=cal_map)

        print("Loading ff training data...")
        ff_train_Xs, ff_train_ys = load_ff_files(ROTAMER_TRAIN_DIR)

        print("Loading ff testing data...")
        ff_test_Xs, ff_test_ys = load_ff_files(ROTAMER_TEST_DIR)


        Xs.extend(ff_train_Xs) # add training data here
        ys.extend(ff_train_ys)

        # shuffle dataset
        Xs, ys = sklearn.utils.shuffle(Xs, ys)

        # subsample_size = int(2e5)
        subsample_size = len(Xs)

        assert subsample_size <= len(Xs)

        Xs, ys = Xs[:subsample_size], ys[:subsample_size]

        print("lengths", len(Xs), len(ys))

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xs, ys, test_size=0.25)

        rd_train = RawDataset(X_train, y_train)
        rd_test  = RawDataset(X_test,  y_test)

        X_gdb11, y_gdb11 = load_hdf5_files([
            os.path.join(ROITBERG_ANI_DIR, "ani1_gdb10_ts.h5"),
        ])

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

    test_l2s = trainer.feed_dataset(sess, fd_test, shuffle=False, target_ops=l2_losses)
 
    best_test_score = np.sqrt(np.mean(np.concatenate(test_l2s).reshape((-1,))))

    max_local_epoch_count = 100

    train_ops = [trainer.global_step, trainer.learning_rate, trainer.rmse, train_op_exp]

    for lr in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        local_epoch_count = 0 # epochs within a learning rate

        print("setting learning rate to", lr)
        sess.run(tf.assign(trainer.learning_rate, lr))
        while local_epoch_count < max_local_epoch_count:

            # sess.run(max_norm_ops) # extraneous, do once at the beginning
            train_results = trainer.feed_dataset(sess, fd_train, shuffle=True, target_ops=train_ops)
            sess.run(max_norm_ops)

            global_epoch = train_results[0][0] // fd_train.num_batches()

            test_l2s = trainer.feed_dataset(sess, fd_test, shuffle=False, target_ops=l2_losses)

            test_rmse = np.sqrt(np.mean(np.concatenate(test_l2s).reshape((-1,))))

            if test_rmse < best_test_score:

                sp = saver.save(sess, save_path, global_step=trainer.global_step)

                gdb11_l2s = trainer.feed_dataset(sess, fd_gdb11, shuffle=False, target_ops=l2_losses)
                gdb11_rmse = np.sqrt(np.mean(np.concatenate(gdb11_l2s).reshape((-1,))))

                fftest_l2s = trainer.feed_dataset(sess, fd_fftest, shuffle=False, target_ops=l2_losses)
                fftest_rmse = np.sqrt(np.mean(np.concatenate(fftest_l2s).reshape((-1,))))

                print("Better g-epo:", global_epoch, "| lr:", lr, \
                      "| l-epo:", local_epoch_count, \
                      "| test rmse:", test_rmse, "| gdb11 rmse:", gdb11_rmse, "| fftest rmse:", fftest_rmse)

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
