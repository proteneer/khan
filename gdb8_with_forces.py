import os
import numpy as np
import time
import tensorflow as tf
import sklearn.model_selection
from sklearn.model_selection import train_test_split

from khan.data.dataset import RawDataset
from khan.training.trainer_multi_tower import TrainerMultiTower, flatten_results, initialize_module
from data_utils import HARTREE_TO_KCAL_PER_MOL
from data_loaders import DataLoader
from concurrent.futures import ThreadPoolExecutor

import multiprocessing
import argparse
import pickle

# import khan.activations

def main():

    parser = argparse.ArgumentParser(description="Run ANI1 neural net training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ani-lib', required=True, help="Location of the shared object for GPU featurization")
    parser.add_argument('--fitted', default=False, action='store_true', help="Whether or use fitted or self-ixn")
    parser.add_argument('--add_ffdata', default=False, action='store_true', help="Whether or not to add the forcefield data")
    parser.add_argument('--gpus', default=1, help="Number of gpus we use")
    parser.add_argument('--train_forces', default=True, help="If we train to the forces")

    parser.add_argument('--save-dir', default='~/work', help="location where save data is dumped")
    parser.add_argument('--train-dir', default='~/ANI-1_release', help="location where work data is dumped")

    args = parser.parse_args()

    print("Arguments", args)

    lib_path = os.path.abspath(args.ani_lib)
    print("Loading custom kernel from", lib_path)
    initialize_module(lib_path)

    ANI_TRAIN_DIR = args.train_dir
    ANI_SAVE_DIR = args.save_dir

    save_dir = os.path.join(ANI_SAVE_DIR, "save")

    use_fitted = args.fitted
    add_ffdata = args.add_ffdata

    data_loader = DataLoader(False)

    # all_Xs, all_Ys = data_loader.load_gdb8(ANI_TRAIN_DIR)

    # todo: ensure disjunction in train_test_valid
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(all_Xs, all_Ys, test_size=0.25) # stratify by UTT would be good to try here
    # rd_train, rd_test = RawDataset(X_train, y_train), RawDataset(X_test,  y_test)


    batch_size = 1024
    X_gdb11, y_gdb11 = data_loader.load_gdb11(ANI_TRAIN_DIR)

    rd_gdb11 = RawDataset(X_gdb11[3*batch_size:], y_gdb11[3*batch_size:])
    config = tf.ConfigProto(allow_soft_placement=True)
    xyf_save = "xyf.pkl"

    if os.path.exists(xyf_save):
        print("-------fast-loading-----")
        xyf = pickle.load(open(xyf_save, 'rb'))
    else:
        xyf = data_loader.load_gdb3_forces("/home/yutong/ANI-1_release/qm") # todo: figure out how to split this consistently later    
        pickle.dump(xyf, open(xyf_save, 'wb'))
    
    all_Xs_f, all_Ys_f, all_Fs_f = xyf

    perm = np.random.permutation(len(all_Ys_f))
    all_Xs_f, all_Ys_f, all_Fs_f = [all_Xs_f[i] for i in perm], [all_Ys_f[i] for i in perm], [all_Fs_f[i] for i in perm]

    frac = int(len(all_Fs_f)*0.75)

    x_train, y_train, f_train = all_Xs_f[:frac], all_Ys_f[:frac], all_Fs_f[:frac]
    x_test, y_test, f_test = all_Xs_f[frac:], all_Ys_f[frac:], all_Fs_f[frac:]

    # x_train, y_train, f_train, x_test, y_test, f_test = train_test_split(all_Xs_f, all_Ys_f, all_Fs_f, test_size=0.25)

    # print(all_Fs_f.shape, f_train.shape)
    # for f in f_train:
        # print(f.shape)

    rd_train_forces = RawDataset(x_train, y_train, f_train)
    rd_test_forces = RawDataset(x_test, y_test)

    with tf.Session(config=config) as sess:

        # This training code implements cross-validation based training, whereby we determine convergence on a given
        # epoch depending on the cross-validation error for a given validation set. When a better cross-validation
        # score is detected, we save the model's parameters as the putative best found parameters. If after more than
        # max_local_epoch_count number of epochs have been run and no progress has been made, we decrease the learning
        # rate and restore the best found parameters.

        n_gpus = int(args.gpus)
        if n_gpus > 0:
            towers = ["/gpu:"+str(i) for i in range(n_gpus)]
        else:
            towers = ["/cpu:"+str(i) for i in range(multiprocessing.cpu_count())]

        print("towers:", towers)

        trainer = TrainerMultiTower(
            sess,
            towers=towers,
            precision=tf.float32,
            layer_sizes=(128, 128, 64, 1),
            activation_fn=tf.nn.relu
        )

        if os.path.exists(save_dir):
            print("Restoring existing model from", save_dir)
            trainer.load(save_dir)
        else:
            trainer.initialize() # initialize to random variables

        # for res in trainer.feed_dataset(
        #     rd_gdb11,
        #     shuffle=False,
        #     target_ops=[trainer.tower_feat_grads],
        #     batch_size=batch_size):
        #     print(type(res))
        #     for r in res[0]:
        #         # print(type(res[0]))
        #         # print(type(r))
        #         print(r)
        #         assert np.any(np.isnan(r)) == False

        all_grads = []
        for nrg in trainer.predict(rd_train_forces):
            # all_grads.append(grad)
            if np.any(np.isnan(nrg)):
                print("FATAL NAN FOUND")
                print(nrg)
                assert 0

        all_grads = []
        for grad in trainer.coordinate_gradients(rd_train_forces):
            if np.any(np.isnan(grad)):
                print("FATAL NAN FOUND")
                print(grad)
                assert 0

        max_local_epoch_count = 10

        train_ops = [
            trainer.global_epoch_count,
            trainer.learning_rate,
            trainer.local_epoch_count,
            trainer.unordered_l2s,
            trainer.train_op,
        ]

        print("------------Starting Training--------------")

        start_time = time.time()

        train_forces = bool(int(args.train_forces)) # python is retarded

        # training with forces
        while sess.run(trainer.learning_rate) > 5e-10: # this is to deal with a numerical error, we technically train to 1e-9

            while sess.run(trainer.local_epoch_count) < max_local_epoch_count:

                start_time = time.time()
                if train_forces:
                    train_results_forces = list(trainer.feed_dataset(
                        rd_train_forces,
                        shuffle=True,
                        target_ops=[trainer.train_op_forces, trainer.tower_force_rmses],
                        batch_size=batch_size,
                        before_hooks=trainer.max_norm_ops))

                print("Force training time", time.time()-start_time)

                start_time = time.time()
                train_results_energies = list(trainer.feed_dataset(
                    rd_train_forces,
                    shuffle=True,
                    target_ops=train_ops,
                    batch_size=batch_size,
                    before_hooks=trainer.max_norm_ops))

                train_abs_rmse = np.sqrt(np.mean(flatten_results(train_results_energies, pos=3))) * HARTREE_TO_KCAL_PER_MOL
                test_abs_rmse = trainer.eval_abs_rmse(rd_test_forces)

                print("Energy training time", time.time()-start_time, "error:", train_abs_rmse, test_abs_rmse)
                # print(time.time()-start_time, train_abs_rmse, test_abs_rmse, gdb11_abs_rmse)


            print("==========Decreasing learning rate==========")
            sess.run(trainer.decr_learning_rate)
            sess.run(trainer.reset_local_epoch_count)
            trainer.load_best_params()

    return



if __name__ == "__main__":
    main()






