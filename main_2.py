import glob
import os
import numpy as np
import tempfile
import time
import tensorflow as tf
import sklearn
import sklearn.model_selection

from khan.training.trainer import Trainer, flatten_results
from khan.training.trainer_multi_gpu import TrainerMultiGPU, flatten_results

from data_utils import HARTREE_TO_KCAL_PER_MOL
from data_loaders import DataLoader
from concurrent.futures import ThreadPoolExecutor

import argparse
import line_profiler
import pyximport
pyximport.install()

import featurizer

def assert_stats(profile, name):
    profile.print_stats()
    stats = profile.get_stats()
    assert len(stats.timings) > 0, "No profile stats."
    for key, timings in stats.timings.items():
        if key[-1] == name:
            assert len(timings) > 0
            break
    else:
        raise ValueError("No stats for %s." % name)

def main():

    parser = argparse.ArgumentParser(description="Run ANI1 neural net training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fitted', default=False, action='store_true', help="Whether or use fitted or self-ixn")
    parser.add_argument('--add_ffdata', default=False, action='store_true', help="Whether or not to add the forcefield data")
    parser.add_argument('--prod', default=False, action='store_true', help="Whether we run over all of gdb8")

    parser.add_argument('--work-dir', default='~/work', help="location where work data is dumped")
    parser.add_argument('--train-dir', default='~/ANI-1_release', help="location where work data is dumped")

    parser.add_argument('--ps', default=False, action='store_true', help="Whether or not we're a parameter server")
    parser.add_argument('--task_idx', default=0,  help="Which data shard the worker will process")

    args = parser.parse_args()

    print("Arguments", args)

    ANI_TRAIN_DIR = args.train_dir
    ANI_WORK_DIR = args.work_dir

    CALIBRATION_FILE_TRAIN = os.path.join(ANI_TRAIN_DIR, "results_QM_M06-2X.txt")
    CALIBRATION_FILE_TEST = os.path.join(ANI_TRAIN_DIR, "gdb_11_cal.txt")
    ROTAMER_TRAIN_DIR = os.path.join(ANI_TRAIN_DIR, "rotamers/train")
    ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "rotamers/test")
    CHARGED_ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "charged_rotamers_2")
    CCSDT_ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "ccsdt_dataset")

    save_dir = os.path.join(ANI_WORK_DIR, "save")
    # data_dir_gdb11 = os.path.join(ANI_WORK_DIR, "gdb11")
    # data_dir_fftest_neutral = os.path.join(ANI_WORK_DIR, "fftest")
    # data_dir_fftest_charged = os.path.join(ANI_WORK_DIR, "fftest_charged")
    # data_dir_fftest_ccsdt = os.path.join(ANI_WORK_DIR, "ccsdt_dataset")

    use_fitted = args.fitted
    add_ffdata = args.add_ffdata

    data_loader = DataLoader(use_fitted)

    if add_ffdata:
        ff_train_dir = ROTAMER_TRAIN_DIR
    else:
        ff_train_dir = None

    rd_train, rd_test = data_loader.load_gdb8(ANI_TRAIN_DIR, CALIBRATION_FILE_TRAIN, ff_train_dir)

    batch_size = 1024
    
    rd_gdb11 = data_loader.load_gdb11(ANI_TRAIN_DIR, CALIBRATION_FILE_TEST)
    rd_ffneutral_mo62x, ffneutral_groups_mo62x = data_loader.load_ff(ROTAMER_TEST_DIR)
    rd_ffneutral_ccsdt, ffneutral_groups_ccsdt = data_loader.load_ff(CCSDT_ROTAMER_TEST_DIR)
    rd_ffcharged_mo62x, ffcharged_groups_mo62x = data_loader.load_ff(CHARGED_ROTAMER_TEST_DIR)

    eval_names    = ["Neutral Rotamers", "Neutral Rotamers CCSDT", "Charged Rotamers"]
    eval_groups   = [ffneutral_groups_mo62x, ffneutral_groups_ccsdt, ffcharged_groups_mo62x]
    eval_datasets = [rd_ffneutral_mo62x, rd_ffneutral_ccsdt, rd_ffcharged_mo62x]

    # eval_names    = ["Neutral Rotamers"]
    # eval_groups   = [ffneutral_groups_mo62x]
    # eval_datasets = [rd_ffneutral_mo62x]

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:

        print("foo")

        trainer = TrainerMultiGPU(sess)
        trainer.initialize()

        l2_losses = [trainer.l2]
        print("Evaluating Rotamer Errors:")

        for name, ff_data, ff_groups in zip(eval_names, eval_datasets, eval_groups):
            print(name, "{0:.6f} kcal/mol".format(trainer.eval_eh_rmse(ff_data, ff_groups)))

        max_local_epoch_count = 100

        train_ops = [
            trainer.global_step,
            trainer.learning_rate,
            trainer.local_epoch_count,
            trainer.l2,
            trainer.train_op
        ]

        best_test_score = trainer.eval_abs_rmse(rd_test)

        print("------------Starting Training--------------")

        start_time = time.time()

        start_epoch = sess.run(trainer.global_step) // rd_train.num_batches(batch_size)

        while sess.run(trainer.learning_rate) > 5e-10:

            while sess.run(trainer.local_epoch_count) < max_local_epoch_count:

                sess.run(trainer.max_norm_ops) # should this run after every batch instead?

                start_time = time.time()
                train_results = trainer.feed_dataset(
                    rd_train,
                    shuffle=True,
                    target_ops=train_ops,
                    batch_size=batch_size)

                global_epoch = train_results[0][0] // rd_train.num_batches(batch_size)
                print("Avg time per epoch", (time.time() - start_time))

                train_abs_rmse = np.sqrt(np.mean(flatten_results(train_results, pos=3))) * HARTREE_TO_KCAL_PER_MOL

                print("train_abs_rmse", train_abs_rmse, "kcal/mol")

                learning_rate = train_results[0][1]
                local_epoch_count = train_results[0][2]

                test_abs_rmse = trainer.eval_abs_rmse(rd_test)
                print(time.strftime("%Y-%m-%d %H:%M:%S"), 'g-epoch', global_epoch, 'l-epoch', local_epoch_count, 'lr', "{0:.0e}".format(learning_rate), \
                    'train abs rmse:', "{0:.2f} kcal/mol,".format(train_abs_rmse), \
                    'test abs rmse:', "{0:.2f} kcal/mol".format(test_abs_rmse), end='')

                if test_abs_rmse < best_test_score:
                    # trainer.save_best_params()
                    gdb11_abs_rmse = trainer.eval_abs_rmse(rd_gdb11)
                    print(' | gdb11 abs rmse', "{0:.2f} kcal/mol | ".format(gdb11_abs_rmse), end='')
                    for name, ff_data, ff_groups in zip(eval_names, eval_datasets, eval_groups):
                        print(name, "abs/rel rmses", "{0:.2f} kcal/mol,".format(trainer.eval_abs_rmse(ff_data)), \
                            "{0:.2f} kcal/mol | ".format(trainer.eval_eh_rmse(ff_data, ff_groups)), end='')

                    best_test_score = test_abs_rmse
                    sess.run(trainer.reset_local_epoch_count)
                else:
                    sess.run(trainer.incr_local_epoch_count)

                # trainer.save(save_dir)

                print('', end='\n')

            print("==========Decreasing learning rate==========")
            sess.run(trainer.decr_learning_rate)
            sess.run(trainer.reset_local_epoch_count)

    return



if __name__ == "__main__":
    main()






