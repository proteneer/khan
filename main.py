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
                ys.append(y)

                X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                Xs.append(X)

            # debug
            # if len(Xs) > 16000:
                # break

    return Xs, ys

if __name__ == "__main__":

    data_dir_train = "/media/yutong/nvme_ssd/v3/train"
    data_dir_test = "/media/yutong/nvme_ssd/v3/test"
    data_dir_gdb11 = "/media/yutong/nvme_ssd/v3/gdb11"


    batch_size = 1024


    if not os.path.exists(data_dir_train):

        Xs, ys = load_hdf5_files([
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s01.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s02.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s03.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s04.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s05.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s06.h5",
            "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s07.h5",
            # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s08.h5",
        ])

        # shuffle dataset
        Xs, ys = sklearn.utils.shuffle(Xs, ys)

        subsample_size = int(1e6)

        assert subsample_size < len(Xs)

        Xs, ys = Xs[:subsample_size], ys[:subsample_size]

        # print(len(Xs), len(ys))

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xs, ys, test_size=0.25)

        rd_train = RawDataset(X_train, y_train)
        rd_test  = RawDataset(X_test,  y_test)

        X_gdb11, y_gdb11 = load_hdf5_files([
            "/home/yutong/roitberg_data/ANI-1_release/ani1_gdb10_ts.h5"
        ])


        rd_gdb11  = RawDataset(X_gdb11,  y_gdb11)

        try:
            os.makedirs(data_dir_train, exist_ok=True)
            os.makedirs(data_dir_test, exist_ok=True)
            os.makedirs(data_dir_gdb11, exist_ok=True)
        except Exception as e:
            print("warning:", e)

        # print("featurizing...")
        fd_train = rd_train.featurize(batch_size, data_dir_train)
        fd_test = rd_test.featurize(batch_size, data_dir_test)
        fd_gdb11 = rd_gdb11.featurize(batch_size, data_dir_gdb11)

    else:

        fd_train = FeaturizedDataset(data_dir_train)
        fd_test = FeaturizedDataset(data_dir_test)
        fd_gdb11 = FeaturizedDataset(data_dir_gdb11)


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

    save_dir = "/media/yutong/nvme_ssd/v3/save"
    # save_dir = "./tmp_save"
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
        print("Pre-minimizing one epoch with rmse loss...")
        # train_ops = [trainer.rmse, train_op_rmse]
        # train_ops.extend(max_norm_ops)
        # train_losses = trainer.feed_dataset(sess, fd_train, shuffle=True, target_ops=train_ops)

    num_epochs = 2
    tot_time = 0
    
    print("num batches", fd_train.num_batches())

    st = time.time()

    # l2_losses = [trainer.l2, trainer.model.predict_op()]

    l2_losses = [trainer.l2]

    test_l2s = trainer.feed_dataset(sess, fd_test, shuffle=False, target_ops=l2_losses)
    
    # print("????", test_l2s[0][1])

    # for a in test_l2s:
    #     for b in a[0]:
    #         if not np.isfinite(b):
    #             print("!!!!", b)




    best_test_score = np.sqrt(np.mean(np.concatenate(test_l2s).reshape((-1,))))

    print("BEST_TEST_SCORE", best_test_score)

    # assert 0

    local_epoch_count = 0 # epochs within a learning rate
    max_local_epoch_count = 50 
    
    # max_local_epoch_count = "test rmse", test_r "gdb11 rmse", 0
    # global_epoch_count = 0

    # use global step?

    # train_ops = [trainer.exp_loss, train_op_exp, trainer.global_step]
    train_ops = [trainer.global_step, trainer.learning_rate, trainer.rmse, train_op_exp]
    # train_ops.extend(max_norm)

    # print("INITIAL WEIGHTS", sess.run(trainer.weight_matrices()))
    # print("initial biases", sess.run(trainer.biases()))
    # assert 0

    for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
    # for lr in [1e-4, 1e-5]:

        print("setting learning rate to", lr)
        sess.run(tf.assign(trainer.learning_rate, lr))

        # print("WTF???", sess.run([trainer.learning_rate]))
        # assert 0

        # for e in range(num_epochs):
        while local_epoch_count < max_local_epoch_count:

            sess.run(max_norm_ops) # extraneous, do once at the beginning
            train_results = trainer.feed_dataset(sess, fd_train, shuffle=True, target_ops=train_ops)
            sess.run(max_norm_ops)

            global_epoch = train_results[0][0] // fd_train.num_batches()

            # print("RMSE DEBUG", train_results[0][2])

            # print(train_results[0][0])
            # global_rmse_T = np.sqrt(np.mean(np.concatenate([x[0] for x in train_losses]).reshape((-1,))))

            test_l2s = trainer.feed_dataset(sess, fd_test, shuffle=False, target_ops=l2_losses)

            # print("L2 MEAN, MODEL PREDS", np.mean(np.concatenate(test_l2s)), test_l2s)

            # for r_idx, r in enumerate(test_l2s):
            #     if not np.isfinite(r).all():
            #         print("WEIGHTS", sess.run(trainer.weight_matrices()))
            #         print("biases", sess.run(trainer.biases()))
            #         assert 0

            #         print(r_idx, r)

            test_rmse = np.sqrt(np.mean(np.concatenate(test_l2s).reshape((-1,))))

            if test_rmse < best_test_score:

                sp = saver.save(sess, save_path, global_step=trainer.global_step)

                gdb11_l2s = trainer.feed_dataset(sess, fd_gdb11, shuffle=False, target_ops=l2_losses)
                gdb11_rmse = np.sqrt(np.mean(np.concatenate(gdb11_l2s).reshape((-1,))))

                print("Better g-epo:", global_epoch, "| lr:", lr, \
                      "| l-epo:", local_epoch_count, \
                      "| test rmse:", test_rmse, "| gdb11 rmse:", gdb11_rmse)

                # print("gdb11 rmse", gdb11_rmse)

                local_epoch_count = 0
            else:

                print("Worse g-epo:", global_epoch, "| lr:", lr, \
                      "| l-epo:", local_epoch_count, \
                      "| test rmse:", test_rmse)
                local_epoch_count += 1




            # global_epoch_count += 1

            # print("train/test/gdb11 rmse", global_rmse_T, global_rmse0, global_rmse1)




        # print("epoch:", e)

        # for fd in [fd_train, fd_test, fd_gdb11]:
        # avg_loss_train = 0
        # for i in range(fd_train.num_batches()):
        #     print("running", i)
        #     # res = sess.run([train_op, loss_op])
        #     # res = sess.run([predict_op, train_op, loss_op])
        #     if e < 36 and idx :
        #         res = sess.run([max_norm, trainer.rmse, train_op_rmse])
        #     else:
        #         res = sess.run([max_norm, trainer.exp_loss, train_op_exp])

        #     avg_batch_rmse = np.mean([x[1] for x in res]) # this isn't strictly the same thing as global rmse
        #     print("avg batch rmse:", avg_batch_rmse)
        # avg_loss_train /= fd_train.num_batches()

        # compute l2s
        # test_l2s = []
        # for i in range(fd_test.num_batches()):
        #     res = sess.run([trainer.l2])
        #     test_l2s.append()

        #     l2s fd_test.num_batches()
        # print(avg_loss_train)

    tot_time = time.time() - st # this logic is a little messed up

    tpm = tot_time/(fd_train.num_batches()*batch_size*num_epochs*3)
    print("Time Per Mol:", tpm, "seconds")
    print("Samples per minute:", 60/tpm)