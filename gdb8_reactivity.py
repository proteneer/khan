import os
import numpy as np
import time
import tensorflow as tf
import sklearn.model_selection
import pickle

from khan.data.dataset import RawDataset
from khan.training.trainer_multi_tower import initialize_module
from khan.training.committee import Committee
from khan.model import activations

from data_utils import load_reactivity_data
from khan.utils.constants import KCAL_MOL_IN_HARTREE
from data_loaders import DataLoader

from khan.lad import lad
from khan.lad import lad_coulomb

from multiprocessing import Pool as ProcessPool 
import multiprocessing
import argparse
import functools
import time

from line_profiler import LineProfiler

PRECISION = {
    "single": tf.float32,
    "double": tf.float64
}

def get_pid(x):
    return os.getpid()

def train_test_split(X, y, test_percent):
    """
    Split data set into train test, handles edge cases
    """

    if test_percent == 1.0:
        X_train = []
        Y_train = [] 
        X_test = list(X)
        Y_test = list(y)
    elif test_percent == 0.0:
        X_train = list(X)
        Y_train = list(y)
        X_test = []
        Y_test = []
    else:
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_percent)

    return X_train, X_test, Y_train, Y_test


def parse_args():
    """
    parse commandline arguments
    """

    parser = argparse.ArgumentParser(
        description="Run ANI1 neural net training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--ani-lib',
        required=True,
        help="Location of the shared object for GPU featurization"
    )

    parser.add_argument(
        '--cpus',
        default=1,
        type=int,
        help="Number of cpus we use"
    )

    parser.add_argument(
        '--save-dir',
        default='~/work',
        help="Location where save data is dumped. If the folder does not exist then it will be created."
    )

    parser.add_argument(
        '--train-dir',
        default='~/ANI-1_release',
        help="Location where training data is located"
    )
    
    parser.add_argument(
        '--test-size',
        default=0.25,
        help='fraction of ANI1 and dimer data in test set'
    )

    parser.add_argument(
        '--lad-data',
        default=None,
        help='Location of reference LAD data (json).  If given long range coulomb energy based on LAD charges will be removed prior to training'
    )

    parser.add_argument(
        '--pickle-output',
        default=None,
        type=str,
        help="if given and doing lad pickle a gdb8 dataset with coulomb removed to this file"
    )

    parser.add_argument(
        '--gdb8-pickle',
        default=None,
        type=str,
        help='pickle file for gdb8 dataset with coulomb removed'
    )

    parser.add_argument(
        '--json-train-dir',
        default=[],
        nargs='*',
        type=str,
        help='location of additional training data (json format)'
    )

    parser.add_argument(
        '--dimer-dir',
        default=None,
        help='location of dimer data'
    )

    parser.add_argument(
        '--dimer-test-percent',
        default=0.25,
        type=float,
        help='percent of dimer data to put in test set'
    )

    parser.add_argument(
        '--reactivity-dir',
        default=None,
        help='location of reactivity data'
    )

    parser.add_argument(
        '--reactivity-test-percent',
        default=0.25,
        type=float,
        help='percent of reactions to put in test set'
    )

    parser.add_argument(
        '--deep-network',
        action='store_true',
        help='Use James super deep network (256, 256, 256, 256, 256, 256, 256, 128, 64, 8, 1)'
    )

    parser.add_argument(
        '--fit-charges',
        action='store_true',
        help='fit charges'
    )

    parser.add_argument(
        '--activation-function',
        type=str,
        choices=activations.get_all_fn_names(),
        help='choice of activation function',
        default="celu"
    )

    parser.add_argument(
        '--precision',
        default='single',
        type=str,
        choices=PRECISION.keys(),
        help="Floating point precision of NN"
    )

    parser.add_argument(
        '--drop-probability',
        default=0.0,
        type=float,
        help='probability of dropping params in dropout'
    )

    parser.add_argument(
        '--ewc-save-file',
        default=None,
        type=str,
        help="*base name* for file to save EWC parameters to (stored in save-dir)"
    )

    parser.add_argument(
        '--ewc-read-file',
        default=None,
        type=str,
        help="*base name* for file which has previously saved EWC parameters to use (assumed to be in save-dir)"
    )

    parser.add_argument(
        '--gdb-min-n',
        default=1,
        type=int,
        help="min gdb dataset to load"
    )

    parser.add_argument(
        '--gdb-max-n',
        default=8,
        type=int,
        help="max gdb dataset to load"
    )

    parser.add_argument(
        '--numb-networks',
        default=1,
        type=int,
        help='number of networks to train'
    )

    args = parser.parse_args()

    if args.ewc_save_file is not None:
        assert not args.ewc_save_file.endswith(".npz")

    print("Input Arguments", args)

    return args


def main():

    args = parse_args()

    # setup lad  
    if args.lad_data:
        lad_params = dict(lad.LAD_PARAMS)
        reference_lads = lad.read_reference_lads(
            args.lad_data, lad_params)

    lib_path = os.path.abspath(args.ani_lib)
    print("Loading custom kernel from", lib_path)
    initialize_module(lib_path)

    ANI_SAVE_DIR = args.save_dir

    # save_dir = os.path.join(ANI_SAVE_DIR, "save")
    save_file = os.path.join(ANI_SAVE_DIR, "save_file.npz")

    data_loader = DataLoader(False)

    ncpu = int(args.cpus)
    pool = None
    if args.cpus > 1:
        print("Using cpu pool with %d processes" % args.cpus)
        pool = ProcessPool(args.cpus)

    if args.gdb8_pickle is not None:
        print("Reading gdb8 pickle")
        with open(args.gdb8_pickle, "rb") as fin:
            rd_tmp = pickle.load(fin)
            all_Xs = rd_tmp.all_Xs
            all_Ys = rd_tmp.all_ys

    else:
        all_Xs, all_Ys = data_loader.load_gdb8(
            args.train_dir, gdb_min_n=args.gdb_min_n, gdb_max_n=args.gdb_max_n)

        if args.lad_data:
            lad_coulomb.remove_coulomb(all_Xs, all_Ys, lad_params, reference_lads, pool)
            if args.pickle_output:
                rd_tmp = RawDataset(all_Xs, all_Ys)
                with open(args.pickle_output, "wb") as fout:
                    pickle.dump(rd_tmp, fout)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        all_Xs, all_Ys, test_size=args.test_size)

    # gdb11
    X_gdb11, y_gdb11 = data_loader.load_gdb11(args.train_dir)
    if args.lad_data:
        lad_coulomb.remove_coulomb(X_gdb11, y_gdb11, lad_params, reference_lads, pool)
    rd_gdb11 = RawDataset(X_gdb11, y_gdb11)

    # extra training data
    if args.json_train_dir:
        for train_dir in args.json_train_dir:

            X_t_all, Y_t_all, _, _ = load_reactivity_data(train_dir, percent_test=0.0)

            if args.lad_data:
                lad_coulomb.remove_coulomb(X_t_all, Y_t_all, lad_params, reference_lads, pool)

            X_t_train, X_t_test, Y_t_train, Y_t_test = train_test_split(
                X_t_all, Y_t_all, 0.0)

            print("Adding %d training datapoints from file %s" % (len(Y_t_train), train_dir))
            print("Adding %d test datapoints from file %s" % (len(Y_t_test), train_dir))

            X_train.extend(X_t_train)
            y_train.extend(Y_t_train)
            X_test.extend(X_t_test)
            y_test.extend(Y_t_test)

    # dimer data
    rd_d_test, rd_d_train = (None, None)
    if args.dimer_dir is not None:

        X_d_all, Y_d_all, _, _ = load_reactivity_data(args.dimer_dir, percent_test=0.0)

        if args.lad_data:
            lad_coulomb.remove_coulomb(X_d_all, Y_d_all, lad_params, reference_lads, pool)

        X_d_train, X_d_test, Y_d_train, Y_d_test = train_test_split(
            X_d_all, Y_d_all, args.dimer_test_percent)

        print("Number of dimer points in training set {0:d}".format(len(Y_d_train)))
        print("Number of dimer points in test set {0:d}".format(len(Y_d_test)))

        # add to training set
        X_train.extend(X_d_train)
        y_train.extend(Y_d_train)
        X_test.extend(X_d_test)
        y_test.extend(Y_d_test)

        rd_d_test = RawDataset(X_d_test, Y_d_test)
        rd_d_train = RawDataset(X_d_train, Y_d_train)

    # reactivity data
    rd_rxn_test, rd_rxn_train, rd_rxn_all  = (None, None, None)
    if args.reactivity_dir is not None:
        # user higher cutoff as rxn is exothermic
        energy_cut = 200.0 / KCAL_MOL_IN_HARTREE
        # add training data
        X_rxn_all, Y_rxn_all, _, _ = load_reactivity_data(
            args.reactivity_dir, percent_test=0.0, energy_cutoff=energy_cut)

        if args.lad_data:
            lad_coulomb.remove_coulomb(X_rxn_all, Y_rxn_all, lad_params, reference_lads, pool)

        X_rxn_train, X_rxn_test, Y_rxn_train, Y_rxn_test = train_test_split(
            X_rxn_all, Y_rxn_all, args.reactivity_test_percent)

        # add to training set
        X_train.extend(X_rxn_train)
        y_train.extend(Y_rxn_train)
        X_test.extend(X_rxn_test)
        y_test.extend(Y_rxn_test)

        print("Number of reactivity points in training set {0:d}".format(len(Y_rxn_train)))
        print("Number of reactivity points in test set {0:d}".format(len(Y_rxn_test)))

        # keep reaction test set separate
        rd_rxn_test = RawDataset(X_rxn_test, Y_rxn_test) if X_rxn_test else None
        rd_rxn_train = RawDataset(X_rxn_train, Y_rxn_train) if X_rxn_train else None

        # redundant, can be eliminated
        rd_rxn_all = RawDataset(X_rxn_test + X_rxn_train, Y_rxn_test + Y_rxn_train)

    # make train/test sets down here after rxn and dimer data have been added
    rd_test = RawDataset(X_test, y_test)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    # suggested by Yutong
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        # This training code implements cross-validation based training, whereby we determine convergence on a given
        # epoch depending on the cross-validation error for a given validation set. When a better cross-validation
        # score is detected, we save the model's parameters as the putative best found parameters. If after more than
        # max_local_epoch_count number of epochs have been run and no progress has been made, we decrease the learning
        # rate and restore the best found parameters.

        #layers = (128, 128, 64, 1)
        # per Adrian 11.16.2018
        layers = (256, 256, 64, 1)
        if args.deep_network:
            layers = (256, 256, 256, 256, 256, 256, 256, 128, 64, 8, 1)

        activation_fn = activations.get_fn_by_name(args.activation_function)
        precision = PRECISION[args.precision]

        committee = Committee(
            args.save_dir,
            args.numb_networks,
            sess,
            precision,
            False, # don't add atomic energy
            args.ewc_read_file,
            activation_fn=activation_fn,
            layer_sizes=layers,
            fit_charges=args.fit_charges,
        )

        # add training data
        committee.add_training_data(X_train, y_train)

        max_local_epoch_count = 10

        best_test_score = committee.eval_abs_rmse(rd_test)
        print("Initial test score %.2f" % best_test_score)

        print("------------Starting Training--------------")

        start_time = time.time()

        while np.mean(committee.run_session_ops(["learning_rate"])) > 1.0e-6:
            while int(np.mean(
                      committee.run_session_ops(["local_epoch_count"])
                  )) < max_local_epoch_count:

                start_time = time.time()
                 
                ave_train_results = committee.train()
                global_epoch = ave_train_results[0]
                local_epoch = ave_train_results[1]
                learning_rate = ave_train_results[2]
                train_abs_rmse = ave_train_results[3]

                time_per_epoch = time.time() - start_time

                test_abs_rmse = committee.eval_abs_rmse(rd_test)

                print(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    'tpe:',
                    "{0:.2f}s,".format(time_per_epoch),
                    'g-epoch',
                    global_epoch,
                    'l-epoch',
                    local_epoch,
                    'lr',
                    "{0:.0e}".format(learning_rate),
                    'train/test abs rmse:',
                    "{0:.2f} kcal/mol,".format(train_abs_rmse),
                    "{0:.2f} kcal/mol".format(test_abs_rmse), end=''
                )

                if test_abs_rmse < best_test_score:
                    gdb11_abs_rmse = committee.eval_abs_rmse(rd_gdb11)
                    print(' | gdb11 abs rmse', "{0:.2f} kcal/mol | ".format(gdb11_abs_rmse), end='')

                    best_test_score = test_abs_rmse

                    _ = committee.run_session_ops(
                        ["incr_global_epoch_count", "reset_local_epoch_count"]
                    )

                    # info about reactivity training
                    rxn_pairs = [
                        (rd_rxn_train, "rxn train"),
                        (rd_rxn_test, "rxn test"),
                        (rd_rxn_all, "rxn all"),
                        (rd_d_test, "dimer test"),
                        (rd_d_train, "dimer train")
                    ]
                    for rd, name in rxn_pairs: 
                        if rd is not None:
                            rxn_abs_rmse = 0.0
                            if rd.num_mols() > 0:
                                rxn_abs_rmse = committee.eval_abs_rmse(rd)
                            
                            print(
                                ' | {0:s} abs rmse '.format(name),
                                "{0:.2f} kcal/mol | ".format(rxn_abs_rmse),
                                end=''
                            )

                else:
                    _ = committee.run_session_ops(
                        ["incr_global_epoch_count", "incr_local_epoch_count"]
                    )

                print('', end='\n')

            print("==========Decreasing learning rate==========")
            _ = committee.run_session_ops(
                ["decr_learning_rate", "reset_local_epoch_count"]
            )

            if args.ewc_save_file:
                commitee.save_ewc_params(args.ewc_save_file)

            committee.save()

    return


if __name__ == "__main__":
    main()

