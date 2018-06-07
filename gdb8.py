import os
import numpy as np
import time
import tensorflow as tf
import sklearn.model_selection

from khan.data.dataset import RawDataset
from khan.training.trainer_multi_tower import TrainerMultiTower, flatten_results, initialize_module
from data_utils import load_reactivity_data, HARTREE_TO_KCAL_PER_MOL, MAX_ATOM_LIMIT
from data_loaders import DataLoader
from concurrent.futures import ThreadPoolExecutor

import multiprocessing
import argparse


def main():

    parser = argparse.ArgumentParser(description="Run ANI1 neural net training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ani_lib', required=True, help="Location of the shared object for GPU featurization")
    parser.add_argument('--fitted', default=False, action='store_true', help="Whether or use fitted or self-ixn")
    parser.add_argument('--add_ffdata', default=False, action='store_true', help="Whether or not to add the forcefield data")
    parser.add_argument('--gpus', default=1, help="Number of gpus we use")

    parser.add_argument('--work-dir', default='~/work', help="location where work data is dumped")
    parser.add_argument('--train-dir', default='~/ANI-1_release', help="location where work data is dumped")
    parser.add_argument('--reactivity-dir', default=None, help='location of reactivity data')
    parser.add_argument('--reactivity-test-percent', default=0.25, type=float, help='percent of reactions to put in test set')

    parser.add_argument('--deep_network', action='store_true', help='Use James super deep network (256, 256, 256, 256, 256, 256, 256, 128, 64, 8, 1)')

    args = parser.parse_args()

    print("Arguments", args)

    lib_path = os.path.abspath(args.ani_lib)
    print("Loading custom kernel from", lib_path)
    initialize_module(lib_path)

    ANI_TRAIN_DIR = args.train_dir
    ANI_WORK_DIR = args.work_dir

    save_dir = os.path.join(ANI_WORK_DIR, "save")

    use_fitted = args.fitted
    add_ffdata = args.add_ffdata

    data_loader = DataLoader(False)

    all_Xs, all_Ys = data_loader.load_gdb8(ANI_TRAIN_DIR)
    

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(all_Xs, all_Ys, test_size=0.25) # stratify by UTT would be good to try here

    rd_rxn = None
    rd_big_rxn = None
    if args.reactivity_dir is not None:
        # add training data
        X_rxn_train, Y_rxn_train, X_rxn_test, Y_rxn_test, X_rxn_big, Y_rxn_big = load_reactivity_data(args.reactivity_dir, args.reactivity_test_percent)

        X_train.extend(X_rxn_train)
        y_train.extend(Y_rxn_train)

        print("Number of reactivity points in training set {0:d}".format(len(Y_rxn_train)))
        print("Number of reactivity points in test set {0:d}".format(len(Y_rxn_test)))

        # keep reaction test set separate
        rd_rxn_test = RawDataset(X_rxn_test, Y_rxn_test) if X_rxn_test else None
        rd_rxn_train = RawDataset(X_rxn_train, Y_rxn_train) if X_rxn_train else None
        # redundant, can be eliminated
        rd_rxn_all = RawDataset(X_rxn_test + X_rxn_train, Y_rxn_test + Y_rxn_train)
        
        # cannot currently handle this in test either
        # everything over 32 atoms
        #rd_big_rxn = RawDataset(X_rxn_big, Y_rxn_big)

    rd_train, rd_test = RawDataset(X_train, y_train), RawDataset(X_test,  y_test)

    X_gdb11, y_gdb11 = data_loader.load_gdb11(ANI_TRAIN_DIR)
    rd_gdb11 = RawDataset(X_gdb11, y_gdb11)

    batch_size = 1024

    config = tf.ConfigProto(allow_soft_placement=True)

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

        layer_sizes = (128, 128, 64, 1)
        if args.deep_network:
            layer_sizes = (256, 256, 256, 256, 256, 256, 256, 128, 64, 8, 1)

        trainer = TrainerMultiTower(
            sess,
            towers=towers,
            layer_sizes=layer_sizes
        )

        if os.path.exists(save_dir):
            print("Restoring existing model from", save_dir)
            trainer.load(save_dir)
        else:
            trainer.initialize() # initialize to random variables

        max_local_epoch_count = 10

        train_ops = [
            trainer.global_epoch_count,
            trainer.learning_rate,
            trainer.local_epoch_count,
            trainer.unordered_l2s,
            trainer.train_op,
        ]

        best_test_score = trainer.eval_abs_rmse(rd_test)

        # Uncomment if you'd like gradients for a dataset
        all_shapes = []
        for feat in trainer.featurize(rd_test):
            all_shapes.append(feat)
        assert len(all_shapes) == rd_test.num_mols()

        all_grads = []
        for grad in trainer.coordinate_gradients(rd_test):
            all_grads.append(grad)
        assert len(all_grads) == rd_test.num_mols()

        print("------------Starting Training--------------")

        start_time = time.time()

        while sess.run(trainer.learning_rate) > 5e-10: # this is to deal with a numerical error, we technically train to 1e-9

            while sess.run(trainer.local_epoch_count) < max_local_epoch_count:

                # sess.run(trainer.max_norm_ops) # should this run after every batch instead?

                start_time = time.time()
                train_results = list(trainer.feed_dataset(
                    rd_train,
                    shuffle=True,
                    target_ops=train_ops,
                    batch_size=batch_size,
                    before_hooks=trainer.max_norm_ops))

                global_epoch = train_results[0][0]
                time_per_epoch = time.time() - start_time
                train_abs_rmse = np.sqrt(np.mean(flatten_results(train_results, pos=3))) * HARTREE_TO_KCAL_PER_MOL
                learning_rate = train_results[0][1]
                local_epoch_count = train_results[0][2]

                test_abs_rmse = trainer.eval_abs_rmse(rd_test)
                print(time.strftime("%Y-%m-%d %H:%M:%S"), 'tpe:', "{0:.2f}s,".format(time_per_epoch), 'g-epoch', global_epoch, 'l-epoch', local_epoch_count, 'lr', "{0:.0e}".format(learning_rate), \
                    'train/test abs rmse:', "{0:.2f} kcal/mol,".format(train_abs_rmse), "{0:.2f} kcal/mol".format(test_abs_rmse), end='')


                if test_abs_rmse < best_test_score:
                    trainer.save_best_params()
                    gdb11_abs_rmse = trainer.eval_abs_rmse(rd_gdb11)
                    print(' | gdb11 abs rmse', "{0:.2f} kcal/mol | ".format(gdb11_abs_rmse), end='')
                    best_test_score = test_abs_rmse

                    for rd, name in [(rd_rxn_train, "train"), (rd_rxn_test, "test"), (rd_rxn_all, "all")]:
                        if rd is not None:
                            rxn_abs_rmse = trainer.eval_abs_rmse(rd)
                            print(' | reactivity abs rmse ({0:s})'.format(name), "{0:.2f} kcal/mol | ".format(rxn_abs_rmse), end='')
                            if name == "test":
                                best_test_score += rxn_abs_rmse


                    sess.run([trainer.incr_global_epoch_count, trainer.reset_local_epoch_count])
                else:
                    sess.run([trainer.incr_global_epoch_count, trainer.incr_local_epoch_count])

                trainer.save(save_dir)

                print('', end='\n')

            print("==========Decreasing learning rate==========")
            sess.run(trainer.decr_learning_rate)
            sess.run(trainer.reset_local_epoch_count)
            trainer.load_best_params()

    return



if __name__ == "__main__":
    main()






