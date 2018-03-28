import glob
import os
import numpy as np
import tempfile
import time
import tensorflow as tf
import sklearn
import sklearn.model_selection


from khan.training.trainer import Trainer, flatten_results

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

    # cluster = tf.train.ClusterSpec({
    #     "ps": ["localhost:5432"],
    #     # "worker": ["localhost:5001", "localhost:5002"]
    #     "worker": ["localhost:5001"]
    # })

    # if args.ps:
    #     job_name = "ps"
    # else:
    #     job_name = "worker"

    args = parser.parse_args()
    print("Arguments", args)

    # server = tf.train.Server(cluster, job_name=job_name, task_index=int(args.task_idx))

    # if args.ps:
        # server.join()
        # sys.exit(0)

    is_chief = int(args.task_idx) == 0

    ANI_TRAIN_DIR = args.train_dir
    ANI_WORK_DIR = args.work_dir

    CALIBRATION_FILE_TRAIN = os.path.join(ANI_TRAIN_DIR, "results_QM_M06-2X.txt")
    CALIBRATION_FILE_TEST = os.path.join(ANI_TRAIN_DIR, "gdb_11_cal.txt")
    ROTAMER_TRAIN_DIR = os.path.join(ANI_TRAIN_DIR, "rotamers/train")
    ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "rotamers/test")
    CHARGED_ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "charged_rotamers_2")
    CCSDT_ROTAMER_TEST_DIR = os.path.join(ANI_TRAIN_DIR, "ccsdt_dataset")

    save_dir = os.path.join(ANI_WORK_DIR, "save")
    data_dir_gdb11 = os.path.join(ANI_WORK_DIR, "gdb11")
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

    rd_train, rd_test = data_loader.load_gdb8(ANI_WORK_DIR, ANI_TRAIN_DIR, CALIBRATION_FILE_TRAIN, ff_train_dir)

    with tf.Session() as sess:

        print("FOO")

        trainer = Trainer.from_mnn_queue(sess)

        target_ops = [trainer.model.predict_op()]

        train_ops = [trainer.global_step, trainer.learning_rate, trainer.local_epoch_count, trainer.l2, trainer.get_train_op_exp()]

        trainer.initialize()

        print("FEED START")

        train_results = trainer.feed_dataset(
            rd_train,
            shuffle=True,
            target_ops=target_ops)



    return


    








    fd_gdb11 = data_loader.load_gdb11(data_dir_gdb11, ANI_TRAIN_DIR, CALIBRATION_FILE_TEST)
    fd_ffneutral_mo62x, ffneutral_groups_mo62x = data_loader.load_ff(os.path.join(ANI_WORK_DIR, "fftest_neutral"), ROTAMER_TEST_DIR)
    fd_ffneutral_ccsdt, ffneutral_groups_ccsdt = data_loader.load_ff(os.path.join(ANI_WORK_DIR, "fftest_neutral_ccsdt"), CCSDT_ROTAMER_TEST_DIR)
    fd_ffcharged_mo62x, ffcharged_groups_mo62x = data_loader.load_ff(os.path.join(ANI_WORK_DIR, "fftest_charged"), CHARGED_ROTAMER_TEST_DIR)

    eval_names    = ["Neutral Rotamers", "Neutral Rotamers CCSDT", "Charged Rotamers"]
    eval_groups   = [ffneutral_groups_mo62x, ffneutral_groups_ccsdt, ffcharged_groups_mo62x]
    eval_datasets = [fd_ffneutral_mo62x, fd_ffneutral_ccsdt, fd_ffcharged_mo62x]

    # sess = tf.Session()

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % int(args.task_idx),
        cluster=cluster)):

        trainer = Trainer.from_mnn_queue()
        train_op_exp = trainer.get_train_op_exp()
        loss_op = trainer.get_loss_op()
        predict_op = trainer.model.predict_op()
        max_norm_ops = trainer.get_maxnorm_ops()

    with tf.train.MonitoredTrainingSession(
        is_chief=is_chief,
        master=server.target,
        save_checkpoint_secs=10) as sess:

        # if os.path.exists(save_dir):
            # print("Loading saved model..")
            # trainer.load(save_dir)
        # else:
        # print("Initializing...")

        trainer.set_session(sess)

        if is_chief:
            trainer.initialize()

        st = time.time()
     
        l2_losses = [trainer.l2]

        print("Evaluating Rotamer Errors:")

        # TODO: perf only if chief

        for name, ff_data, ff_groups in zip(eval_names, eval_datasets, eval_groups):
            print(name, "{0:.2f} kcal/mol".format(trainer.eval_eh_rmse(ff_data, ff_groups)))

        max_local_epoch_count = 100

        train_ops = [trainer.global_step, trainer.learning_rate, trainer.local_epoch_count, trainer.l2, train_op_exp]

        best_test_score = trainer.eval_abs_rmse(fd_test)

 
        print("------------Starting Training--------------")

        start_time = time.time()

        start_epoch = sess.run(trainer.global_step) // fd_train.num_batches()

        while sess.run(trainer.learning_rate) > 5e-10:

            while sess.run(trainer.local_epoch_count) < max_local_epoch_count:

                sess.run(max_norm_ops)
                train_results = trainer.feed_dataset(
                    fd_train,
                    shuffle=True,
                    target_ops=train_ops)

                global_epoch = train_results[0][0] // fd_train.num_batches()
                print("Avg time per epoch", (time.time() - start_time) / (global_epoch - start_epoch))
                # start_time = time.time()

                ## DEBUG CONTINUE

                # continue
                ## DEBUG

                train_abs_rmse = np.sqrt(np.mean(flatten_results(train_results, pos=3))) * HARTREE_TO_KCAL_PER_MOL

                learning_rate = train_results[0][1]
                local_epoch_count = train_results[0][2]

                test_abs_rmse = trainer.eval_abs_rmse(fd_test)
                print(time.strftime("%Y-%m-%d %H:%M"), 'g-epoch', global_epoch, 'l-epoch', local_epoch_count, 'lr', "{0:.0e}".format(learning_rate), \
                    'train abs rmse:', "{0:.2f} kcal/mol,".format(train_abs_rmse), \
                    'test abs rmse:', "{0:.2f} kcal/mol".format(test_abs_rmse), end='')

                if test_abs_rmse < best_test_score:
                    trainer.save_best_params()
                    gdb11_abs_rmse = trainer.eval_abs_rmse(fd_gdb11)
                    print(' | gdb11 abs rmse', "{0:.2f} kcal/mol | ".format(gdb11_abs_rmse), end='')
                    for name, ff_data, ff_groups in zip(eval_names, eval_datasets, eval_groups):
                        print(name, "abs/rel rmses", "{0:.2f} kcal/mol,".format(trainer.eval_abs_rmse(ff_data)), \
                            "{0:.2f} kcal/mol | ".format(trainer.eval_eh_rmse(ff_data, ff_groups)), end='')

                    # local_epoch_count = 0
                    best_test_score = test_abs_rmse
                    sess.run(trainer.reset_local_epoch_count)
                else:
                    sess.run(trainer.incr_local_epoch_count)

                # trainer.save(save_dir)

                print('', end='\n')

            sess.run(trainer.decr_learning_rate)
            sess.run(trainer.reset_local_epoch_count)
            trainer.load_best_params()

        # print("DECR RESULT", sess.run(trainer.learning_rate))


    # tot_time = time.time() - st # this logic is a little messed up

    # tpm = tot_time/(fd_train.num_batches()*batch_size*num_epochs*3)
    # print("Time Per Mol:", tpm, "seconds")
    # print("Samples per minute:", 60/tpm)

if __name__ == "__main__":

    main()

    # lp = line_profiler.LineProfiler(featurizer.ANI1)
    # X_feat = np.zeros((384, ), dtype=np.float32)
    # lp.runcall(featurizer.ANI1, np.array([[1,2,3],[4,5,6]], dtype=np.float32), np.array([0, 1], dtype=np.int32), X_feat)
    # lp.print_stats()

    # lp.add_function(featurizer.ANI1)
    # lp_wrapper = lp(main)
    # lp_wrapper()
    # lp.print_stats()