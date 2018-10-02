import functools
import os
import numpy as np
import time
import tensorflow as tf
import sklearn.model_selection

from khan.data.dataset import RawDataset
from khan.training.trainer_multi_tower import TrainerMultiTower, flatten_results, initialize_module
from data_utils import HARTREE_TO_KCAL_PER_MOL
from data_loaders import DataLoader
from concurrent.futures import ThreadPoolExecutor

from multiprocessing.dummy import Pool as ThreadPool 

import multiprocessing
import argparse



def expected_information_gain(trainers):
    model_energies = []
    model_x0s = []

    for t in trainers:
        print("Computing gain from", t)
        model_energies.append(t.tower_preds[0])
        x, y, z, m = t.tower_xs[0], t.tower_ys[0], t.tower_zs[0], t.tower_ms[0]
        x0 = tf.stack([x,y,z])
        model_x0s.append(x0)

    expected_info_gain = 0.0

    # (ytz): implement custom information KL/Prob stuff here
    for e, x in zip(model_energies, model_x0s):
        expected_info_gain += tf.norm(e)
        expected_info_gain += tf.norm(x)

    model_gradients = []

    for x in model_x0s:
        grad = tf.gradients(expected_info_gain, x)
        model_gradients.append(grad)

    return expected_info_gain, model_gradients


# def run_one_epoch(args):

#     trainer, rd_train, rd_test = args

#     # predicted energies
#     energies = trainer.tower_preds[0]

#     train_ops = [
#         trainer.global_epoch_count,
#         trainer.learning_rate,
#         trainer.local_epoch_count,
#         trainer.unordered_l2s,
#         trainer.train_op,
#     ]

#     batch_size = 1024

#     start_time = time.time()
#     train_results = list(trainer.feed_dataset(
#         rd_train,
#         shuffle=True,
#         target_ops=train_ops,
#         batch_size=batch_size,
#         before_hooks=trainer.max_norm_ops))

#     global_epoch = train_results[0][0]
#     time_per_epoch = time.time() - start_time
#     train_abs_rmse = np.sqrt(np.mean(flatten_results(train_results, pos=3))) * HARTREE_TO_KCAL_PER_MOL
#     learning_rate = train_results[0][1]
#     local_epoch_count = train_results[0][2]

#     test_abs_rmse = trainer.eval_abs_rmse(rd_test)
#     print("trainer:", trainer, time.strftime("%Y-%m-%d %H:%M:%S"), 'tpe:', "{0:.2f}s,".format(time_per_epoch), 'g-epoch', global_epoch, 'l-epoch', local_epoch_count, 'lr', "{0:.0e}".format(learning_rate), \
#         'train/test abs rmse:', "{0:.2f} kcal/mol,".format(train_abs_rmse), "{0:.2f} kcal/mol".format(test_abs_rmse), end='\n')


def generate_feed_dict(trainer, mol_xs, mol_idxs, mol_yts, mol_grads, b_idx):

    atom_types = (mol_xs[:, 0]).astype(np.int32)

    feed_dict = {
        trainer.x_enq: mol_xs[:, 1],
        trainer.y_enq: mol_xs[:, 2],
        trainer.z_enq: mol_xs[:, 3],
        trainer.a_enq: atom_types,
        trainer.m_enq: mol_idxs,
        trainer.yt_enq: mol_yts,
        trainer.bi_enq: b_idx
    }

    if mol_grads is not None:
        feed_dict[trainer.force_enq_x] = mol_grads[:, 0]
        feed_dict[trainer.force_enq_y] = mol_grads[:, 1]
        feed_dict[trainer.force_enq_z] = mol_grads[:, 2]
    else:
        num_mols = mol_xs.shape[0]
        feed_dict[trainer.force_enq_x] = np.zeros((num_mols, 0), dtype=trainer.precision.as_numpy_dtype)
        feed_dict[trainer.force_enq_y] = np.zeros((num_mols, 0), dtype=trainer.precision.as_numpy_dtype)
        feed_dict[trainer.force_enq_z] = np.zeros((num_mols, 0), dtype=trainer.precision.as_numpy_dtype)

    return feed_dict


def submit_dataset(session, trainers, dataset, batch_size, shuffle, fuzz=None):

    accum = 0

    try:

        executor = ThreadPoolExecutor(len(trainers))
        
        futures = []
        n_batches = dataset.num_batches(batch_size)
        for b_idx, (mol_xs, mol_idxs, mol_yts, mol_grads) in enumerate(dataset.iterate(batch_size=batch_size, shuffle=shuffle, fuzz=fuzz)):
            
            atom_types = (mol_xs[:, 0]).astype(np.int32)

            for t_idx, trainer in enumerate(trainers):

                assert(trainer.num_towers == 1)

                # def closure():
                #     feed_dict = {
                #         trainer.x_enq: mol_xs[:, 1],
                #         trainer.y_enq: mol_xs[:, 2],
                #         trainer.z_enq: mol_xs[:, 3],
                #         trainer.a_enq: atom_types,
                #         trainer.m_enq: mol_idxs,
                #         trainer.yt_enq: mol_yts,
                #         trainer.bi_enq: b_idx
                #     }

                #     if mol_grads is not None:
                #         feed_dict[trainer.force_enq_x] = mol_grads[:, 0]
                #         feed_dict[trainer.force_enq_y] = mol_grads[:, 1]
                #         feed_dict[trainer.force_enq_z] = mol_grads[:, 2]
                #     else:
                #         num_mols = mol_xs.shape[0]
                #         feed_dict[trainer.force_enq_x] = np.zeros((num_mols, 0), dtype=trainer.precision.as_numpy_dtype)
                #         feed_dict[trainer.force_enq_y] = np.zeros((num_mols, 0), dtype=trainer.precision.as_numpy_dtype)
                #         feed_dict[trainer.force_enq_z] = np.zeros((num_mols, 0), dtype=trainer.precision.as_numpy_dtype)


                #     print("Enqueue", b_idx, "on trainer", t_idx)
                #     print("Feed_dict", feed_dict)
                fd = generate_feed_dict(trainer, mol_xs, mol_idxs, mol_yts, mol_grads, b_idx)
                future = executor.submit(functools.partial(session.run, fetches=trainer.put_op, feed_dict=fd))

                futures.append(future)

            print("waiting...")
            # important in conjunction with underlying FIFO queue to ensure order
            for f in futures:
                f.result()

    except Exception as e:
        print("QueueError:", e)
        exit()



def main():

    parser = argparse.ArgumentParser(description="Run ANI1 neural net training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ani_lib', required=True, help="Location of the shared object for GPU featurization")
    parser.add_argument('--fitted', default=False, action='store_true', help="Whether or use fitted or self-ixn")
    parser.add_argument('--add_ffdata', default=False, action='store_true', help="Whether or not to add the forcefield data")
    parser.add_argument('--gpus', default=1, help="Number of gpus we use")

    parser.add_argument('--work-dir', default='~/work', help="location where work data is dumped")
    parser.add_argument('--train-dir', default='~/ANI-1_release', help="location where work data is dumped")

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
    rd_train, rd_test = RawDataset(X_train, y_train), RawDataset(X_test,  y_test)

    X_gdb11, y_gdb11 = data_loader.load_gdb11(ANI_TRAIN_DIR)
    rd_gdb11 = RawDataset(X_gdb11, y_gdb11)

    batch_size = 4

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:

        # This training code implements cross-validation based training, whereby we determine convergence on a given
        # epoch depending on the cross-validation error for a given validation set. When a better cross-validation
        # score is detected, we save the model's parameters as the putative best found parameters. If after more than
        # max_local_epoch_count number of epochs have been run and no progress has been made, we decrease the learning
        # rate and restore the best found parameters.

        # n_gpus = int(args.gpus)
        # if n_gpus > 0:
        #     towers = ["/gpu:"+str(i) for i in range(n_gpus)]
        # else:
        #     towers = ["/cpu:"+str(i) for i in range(multiprocessing.cpu_count())]

        # print("towers:", towers)


        M = 2

        trainers = []

        for m in range(M):
            with tf.variable_scope("model_"+str(m)):
                trainer = TrainerMultiTower(
                    sess,
                    towers=["/cpu:"+str(m)],
                    layer_sizes=(128, 128, 64, 1),
                    precision=tf.float32
                )
                trainers.append(trainer)

        expected_gain, model_gradients = expected_information_gain(trainers)

        sess.run(tf.global_variables_initializer())


        # submit_dataset(session=sess, trainers=trainers, dataset=rd_train, batch_size=batch_size, shuffle=True, fuzz=None)

        # assert 0

        executor = ThreadPoolExecutor(1)

        # executor.submit(functools.partial(trainer.submit_dataset, dataset=rd_train, batch_size=batch_size, shuffle=True))

        future = executor.submit(submit_dataset, session=sess, trainers=trainers, dataset=rd_train, batch_size=batch_size, shuffle=True, fuzz=None)


        print("Waiting...")


        n_batches = rd_train.num_batches(batch_size)

        for bid in range(n_batches):
            print("Dequeue", bid)
            sess.run([expected_gain, model_gradients])
            print("Dequeue done")
        
        future.result()

        # time.sleep(10)

        assert 0




        # sess.run(tf.global_variables_initializer())

        pool = ThreadPool(2)

        for e in range(5):
            pool.map(run_one_epoch, all_data)

        # need to use saver across all
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "model.ckpt")
        saver.save(sess, save_path)


    #     print("all_vars", tf.trainable_variables())

    #     if os.path.exists(save_dir):
    #         print("Restoring existing model from", save_dir)
    #         trainer.load(save_dir)
    #     else:


    #     max_local_epoch_count = 100

    #     train_ops = [
    #         trainer.global_epoch_count,
    #         trainer.learning_rate,
    #         trainer.local_epoch_count,
    #         trainer.unordered_l2s,
    #         trainer.train_op,
    #     ]

    #     best_test_score = trainer.eval_abs_rmse(rd_test)

    #     # Uncomment if you'd like gradients for a dataset
    #     # for feat in trainer.featurize(rd_test):
    #     #     print(feat.shape)

    #     # for grad in trainer.coordinate_gradients(rd_test):
    #     #     print(grad.shape)

    #     print("------------Starting Training--------------")

    #     start_time = time.time()

    #     while sess.run(trainer.learning_rate) > 5e-10: # this is to deal with a numerical error, we technically train to 1e-9

    #         while sess.run(trainer.local_epoch_count) < max_local_epoch_count:

    #             # sess.run(trainer.max_norm_ops) # should this run after every batch instead?

    #             start_time = time.time()
    #             train_results = list(trainer.feed_dataset(
    #                 rd_train,
    #                 shuffle=True,
    #                 target_ops=train_ops,
    #                 batch_size=batch_size,
    #                 before_hooks=trainer.max_norm_ops))

    #             global_epoch = train_results[0][0]
    #             time_per_epoch = time.time() - start_time
    #             train_abs_rmse = np.sqrt(np.mean(flatten_results(train_results, pos=3))) * HARTREE_TO_KCAL_PER_MOL
    #             learning_rate = train_results[0][1]
    #             local_epoch_count = train_results[0][2]

    #             test_abs_rmse = trainer.eval_abs_rmse(rd_test)
    #             print(time.strftime("%Y-%m-%d %H:%M:%S"), 'tpe:', "{0:.2f}s,".format(time_per_epoch), 'g-epoch', global_epoch, 'l-epoch', local_epoch_count, 'lr', "{0:.0e}".format(learning_rate), \
    #                 'train/test abs rmse:', "{0:.2f} kcal/mol,".format(train_abs_rmse), "{0:.2f} kcal/mol".format(test_abs_rmse), end='')

    #             if test_abs_rmse < best_test_score:
    #                 trainer.save_best_params()
    #                 gdb11_abs_rmse = trainer.eval_abs_rmse(rd_gdb11)
    #                 print(' | gdb11 abs rmse', "{0:.2f} kcal/mol | ".format(gdb11_abs_rmse), end='')

    #                 best_test_score = test_abs_rmse
    #                 sess.run([trainer.incr_global_epoch_count, trainer.reset_local_epoch_count])
    #             else:
    #                 sess.run([trainer.incr_global_epoch_count, trainer.incr_local_epoch_count])

    #             trainer.save(save_dir)

    #             print('', end='\n')

    #         print("==========Decreasing learning rate==========")
    #         sess.run(trainer.decr_learning_rate)
    #         sess.run(trainer.reset_local_epoch_count)
    #         trainer.load_best_params()

    # return



if __name__ == "__main__":
    main()






