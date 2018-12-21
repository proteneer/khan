import os
import numpy as np
import time
import pickle
import random
import tensorflow as tf
from tensorflow.python.client import device_lib

import sklearn.model_selection

from khan.data.dataset import RawDataset
from khan.training.trainer_multi_tower import TrainerMultiTower, flatten_results, initialize_module

from data_loaders import DataLoader
from concurrent.futures import ThreadPoolExecutor

import argparse


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def apply_nonbonded_correction(dataset):
    emin = 0.0003 # Hartree. Positive because we're applying it to the dataset: would be negative if applied to the model. 
    for mol_i in range(dataset.num_mols()):
        nonbonded_approx = 0.0
        xx = dataset.all_Xs[mol_i]
        for i, x1 in enumerate(xx):
            if x1[0] == 0.0: continue # skip H
            for x2 in xx[i+1:]:
                if x2[0] == 0.0: continue # skip H
                r2 = np.sum( (x1[1:] - x2[1:])**2 )
                nonbonded_approx += emin * 5**6/(r2**6 + 5**12)**0.5
            #print(x1[0], x2[0], 0.1 * 5**6/(r2**6 + 5**12)**0.5 )
        dataset.all_ys[mol_i] += nonbonded_approx
    print('Nonbonded correction of dataset size', dataset.num_mols())

def main():
    #avail_gpus = get_available_gpus()
    #print("Available GPUs:", avail_gpus)

    print('os.environ:', os.environ)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess: # must be at start to reserve GPUs

        parser = argparse.ArgumentParser(description="Run ANI1 neural net training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--ani_lib', required=True, help="Location of the shared object for GPU featurization")
        parser.add_argument('--fitted', default=False, action='store_true', help="Whether or use fitted energy corrections")
        parser.add_argument('--add_ffdata', default=True, action='store_true', help="Whether or not to add the forcefield data")
        parser.add_argument('--gpus', default='4', help="Number of GPUs to use")
        parser.add_argument('--cpus', default='1', help="Number of CPUs to use (GPUs override this if > 0)")
        parser.add_argument('--start_batch_size', default='64', help="How many training points to consider before calculating each gradient")
        parser.add_argument('--max_local_epoch_count', default='50', help="How many epochs to try each learning rate before reducing it")
        parser.add_argument('--dataset_index', default='0', help="Index of training set to use")
        parser.add_argument('--testset_index', default='0', help="Index of test set to use")
        parser.add_argument('--fit_charges', default=False, action='store_true', help="Whether or not to add fitted charge energies")

        parser.add_argument('--work-dir', default='~/work', help="location where work data is dumped")
        parser.add_argument('--train-dir', default='/home/yzhao/ANI-1_release', help="location where work data is dumped")
        parser.add_argument('--restart', default=False, action='store_true', help="Whether to restart from the save dir")
        parser.add_argument('--train_size', default='0.5', help="how much of the dataset to use for gradient evaluations")
        parser.add_argument('--test_size', default='0.5', help="how much of the dataset to use for testing the energies")

        args = parser.parse_args()

        print("Arguments", args)

        lib_path = os.path.abspath(args.ani_lib)
        print("Loading custom kernel from", lib_path)
        initialize_module(lib_path)

        ANI_TRAIN_DIR = args.train_dir
        ANI_WORK_DIR = args.work_dir
        GRAPH_DB_TRAIN_DIR = '/nfs/working/scidev/stevenso/learning/khan/graphdb_xyz/xyz/train'
        GRAPH_DB_TEST_DIR = '/nfs/working/scidev/stevenso/learning/khan/graphdb_xyz/xyz/test/'
        train_size = float(args.train_size)
        test_size = float(args.test_size)

        CALIBRATION_FILE_TRAIN = os.path.join(ANI_TRAIN_DIR, "results_QM_M06-2X.txt")
        CALIBRATION_FILE_TEST = os.path.join(ANI_TRAIN_DIR, "gdb_11_cal.txt")
        ROTAMER_TRAIN_DIR = [ os.path.join(ANI_TRAIN_DIR, "rotamers/train"), os.path.join(ANI_TRAIN_DIR, "rotamers/test") ]
        ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "rotamers/test")
        CHARGED_ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "charged_rotamers_2")
        CCSDT_ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "ccsdt_dataset")

        save_dir = os.path.join(ANI_WORK_DIR, "save")
        if os.path.isdir(save_dir) and not args.restart:
            print('save_dir', save_dir, 'exists and this is not a restart job')
            exit()
        batch_size = int(args.start_batch_size)
        use_fitted = args.fitted
        add_ffdata = args.add_ffdata
        data_loader = DataLoader(use_fitted)

        print("------------Load evaluation data--------------")
        
        pickle_files = ['eval_new_graphdb.pickle', 'eval_data_old_fftest.pickle', 'eval_data_graphdb.pickle', 'rotamer_gdb_opt.pickle']
        pickle_file = pickle_files[ int(args.testset_index) ]
        if os.path.isfile(pickle_file):
            print('Loading pickle from', pickle_file)
            rd_gdb11, rd_ffneutral_mo62x, ffneutral_groups_mo62x, rd_ffneutral_ccsdt, ffneutral_groups_ccsdt, rd_ffcharged_mo62x, ffcharged_groups_mo62x, rd_gdb_opt, gdb_opt_groups = pickle.load( open(pickle_file, "rb") )
        else:
            print('Loading data fresh (may take some time)')
            xs, ys = data_loader.load_gdb11(ANI_TRAIN_DIR, CALIBRATION_FILE_TEST)
            rd_gdb11 = RawDataset(xs, ys)
            xs, ys, ffneutral_groups_mo62x = data_loader.load_ff(GRAPH_DB_TEST_DIR)     
            rd_ffneutral_mo62x = RawDataset(xs, ys)
            xs, ys, ffneutral_groups_ccsdt = data_loader.load_ff(CCSDT_ROTAMER_TEST_DIR)
            rd_ffneutral_ccsdt = RawDataset(xs, ys)
            xs, ys, ffcharged_groups_mo62x =  data_loader.load_ff(CHARGED_ROTAMER_TEST_DIR)
            rd_ffcharged_mo62x = RawDataset(xs, ys)
            xs, ys, gdb_opt_groups = data_loader.load_ff('haoyu_opt/xyz/')
            rd_gdb_opt = RawDataset(xs, ys)
            print('Pickling data...')
            pickle.dump( (rd_gdb11,
                          rd_ffneutral_mo62x, ffneutral_groups_mo62x,
                          rd_ffneutral_ccsdt, ffneutral_groups_ccsdt, 
                          rd_ffcharged_mo62x, ffcharged_groups_mo62x,
                          rd_gdb_opt, gdb_opt_groups), open( pickle_file, "wb" ) )

        eval_names    = ["Neutral Rotamers", "Neutral Rotamers CCSDT", "Charged Rotamers", "GDB Opt"]
        eval_groups   = [ffneutral_groups_mo62x, ffneutral_groups_ccsdt, ffcharged_groups_mo62x, gdb_opt_groups]
        eval_datasets = [rd_ffneutral_mo62x, rd_ffneutral_ccsdt, rd_ffcharged_mo62x, rd_gdb_opt]

        # This training code implements cross-validation based training, whereby we determine convergence on a given
        # epoch depending on the cross-validation error for a given validation set. When a better cross-validation
        # score is detected, we save the model's parameters as the putative best found parameters. If after more than
        # max_local_epoch_count number of epochs have been run and no progress has been made, we decrease the learning
        # rate and restore the best found parameters.

        max_local_epoch_count = int(args.max_local_epoch_count)
        n_gpus = int(args.gpus) # min( int(args.gpus), len(avail_gpus) )
        n_cpus = min( int(args.cpus), os.cpu_count() )
        if n_gpus > 0:
            towers = ["/gpu:"+str(i) for i in range(n_gpus)]
        else:
            towers = ["/cpu:"+str(i) for i in range(n_cpus)]

        print("towers:", towers)

        #layer_sizes=(128, 128, 64, 1) # original
        layer_sizes=tuple( 6*[256] + [1] )
        #layer_sizes=(1,) # linear
        print('layer_sizes:', layer_sizes)
        n_weights = sum( [layer_sizes[i]*layer_sizes[i+1] for i in range(len(layer_sizes)-1)] )
        print('n_weights:', n_weights)

        print("------------Load training data--------------")
        
        pickle_files = ["gdb8_fftrain_fftest_xy.pickle", "gdb8_graphdb_xy.pickle", "gdb8_xy.pickle", "gdb7_xy.pickle", "gdb6_ffdata_xy.pickle", "gdb3_xy.pickle", "gdb8_graphdb_xy_differ3.pickle"]
        pickle_file = pickle_files[ int(args.dataset_index) ]
        if os.path.isfile(pickle_file):
            print('Loading pickle from', pickle_file)
            Xs, ys = pickle.load( open(pickle_file, "rb") )
        else:
            ff_train_dirs = ROTAMER_TRAIN_DIR + [GRAPH_DB_TRAIN_DIR]
            Xs, ys = data_loader.load_gdb8(ANI_TRAIN_DIR, CALIBRATION_FILE_TRAIN, ff_train_dirs)
            print('Pickling data...')
            pickle.dump( (Xs, ys), open( pickle_file, "wb" ) )

        print("------------Initializing model--------------")

        #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xs, ys, train_size=train_size, test_size=test_size)

        # split train/test by unique element tuples
        data = zip(Xs, ys)
        data_by_elements = {}
        for x, y in data:
            elements = tuple(x[0,:]) # consider sorting this
            if elements not in data_by_elements:
                data_by_elements[elements] = []
            data_by_elements[elements].append( (x, y) )
        all_elements = list(data_by_elements.keys())
        random.shuffle(all_elements)
        n_train = int(train_size * len(Xs))
        n_test = int(test_size * len(Xs))
        X_train, X_test, y_train, y_test = [], [], [], []
        for elements in all_elements:
            xy = data_by_elements[elements]
            xx, yy = zip(*xy)
            if len(y_train) < n_train:
                X_train += xx
                y_train += yy
            elif len(y_test) < n_test:
                X_test += xx
                y_test += yy
            else:
                break

        rd_train, rd_test = RawDataset(X_train, y_train), RawDataset(X_test,

  y_test)
        print( 'n_train =', len(y_train), 'n_test =', len(y_test) )

        trainer = TrainerMultiTower(
            sess,
            towers=towers,
            layer_sizes=layer_sizes,
            fit_charges=args.fit_charges,
            precision=tf.float32 # train in single precision (james you may want to change this later)
            #precision=tf.float64
        )

        if os.path.exists(save_dir):
            print("Restoring existing model from", save_dir)
            trainer.load_numpy(save_dir+'/best.npz')
        else: # initialize new random parameters
            trainer.initialize()

        if False:  # basic nonbonded correction term
            for dataset in ([rd_train, rd_test, rd_gdb11] + eval_datasets[:-1]):
                apply_nonbonded_correction(dataset)

        for name, ff_data, ff_groups in zip(eval_names[:-1], eval_datasets[:-1], eval_groups[:-1]):
            print(name, "abs/rel rmses: {0:.6f} kcal/mol | ".format(trainer.eval_abs_rmse(ff_data)) + "{0:.6f} kcal/mol".format(trainer.eval_eh_rmse(ff_data, ff_groups))) 

        print("------------Starting Training--------------")
        trainer.train(save_dir, rd_train, rd_test, rd_gdb11, eval_names[:1], eval_datasets[:1], eval_groups[:1], batch_size, max_local_epoch_count)

if __name__ == "__main__":
    main()

