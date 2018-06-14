# KHAN: open-source deep-learning package for molecular energies	

As part of Schrodinger's machine-learning collaboration with the Pande lab and the DeepChem project,
we've taken the initial pure tensorflow implementation written by Michael Wu (@miaecle) and optimized
it significantly to reduce the training time.  The original ANI-1 by Smith et al. (doi:10.1039/C6SC05720A)
unfortunately did not provide the training algorithm that was used to generate the released models. 

Note: that this code will at some point be merged back upstream with the DeepChem master, once the
build procedures have been streamlined.

## Features

1. A friendly tensorflow environment with custom ops for performance
2. On-the-fly GPU featurization 
3. Simple to use python API for training
4. Highly extensible with additional datasets
5. Scalable to multiple gpus on a single node
6. Ability to train to forces
7. Double precision for precise optimization routines

## Performance

The code is designed to scale pseudo-linearly with the number of GPUs. As an example, 
under the following configuration:

```
# ani_op.cu.cc

MAX_ATOM_TYPES = 4;

NUM_R_Rs = 16;
R_eta = 16;
R_Rc = 4.6;

A_Rc = 3.1;
A_eta = 6.0;
A_zeta = 8.0;
NUM_A_THETAS = 8;
NUM_A_RS = 4;

TOTAL_FEATURE_SIZE = 384

# main_2.py

batch_size = 1024
train_size/epoch_size = 0.75 * 22 million (training set size of gdb8 under a 75:25 split) conformations

num_gpus = 8
```

On 8x GTX-1080s Tis, we it takes on average about 56 seconds per epoch, or about 1542 epochs/day if doing
pure training. A typical ANI-1 model with a scaled learning rate of 1e-3 to 1e-9 would take about ~2000
epochs to converge. However, while training on multi GPUs leads to faster epoch times, it may not necessarily
lead to faster training time due to gradient averaging.

## Installation Procedures

Requirements:

- Linux
- Python 3.x
- CUDA-enabled GPU
- Modern C++11 compatible gcc

Refer to the Makefile for build instructions. The TF_CFLAGS and TF_LFLAGS can be found according to:

``` bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
```

This will generate an ani.so file that includes both the sorting routing as well as the featurization.

## Example Run

After downloading the gdb8 and gdb10 datasets in the following section, you can run the example code with:

``` bash
python gdb8.py --train-dir ANI_DATA_DIR --work-dir /tmp/model/ --gpus  --ani
```

The output should look something like:
``` bash
2018-04-05 15:31:45.949197: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1423] Adding visible gpu devices: 0, 1, 2
2018-04-05 15:31:46.872747: I tensorflow/core/common_runtime/gpu/gpu_device.cc:911] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-04-05 15:31:46.872785: I tensorflow/core/common_runtime/gpu/gpu_device.cc:917]      0 1 2 
2018-04-05 15:31:46.872794: I tensorflow/core/common_runtime/gpu/gpu_device.cc:930] 0:   N N N 
2018-04-05 15:31:46.872800: I tensorflow/core/common_runtime/gpu/gpu_device.cc:930] 1:   N N Y 
2018-04-05 15:31:46.872806: I tensorflow/core/common_runtime/gpu/gpu_device.cc:930] 2:   N Y N 
2018-04-05 15:31:46.875790: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11380 MB memory) -> physical GPU (device: 0, name: TITAN X (Pascal), pci bus id: 0000:05:00.0, compute capability: 6.1)
2018-04-05 15:31:47.055074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 7543 MB memory) -> physical GPU (device: 1, name: GeForce GTX 1080, pci bus id: 0000:02:00.0, compute capability: 6.1)
2018-04-05 15:31:47.172668: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 6209 MB memory) -> physical GPU (device: 2, name: GeForce GTX 1080, pci bus id: 0000:4f:00.0, compute capability: 6.1)
Evaluating Rotamer Errors:
------------Starting Training--------------
2018-04-05 15:33:29 tpe: 68.03s, g-epoch 0 l-epoch 0 lr 1e-03 train/test abs rmse: 52.67 kcal/mol, 28.59 kcal/mol | gdb11 abs rmse 45.49 kcal/mol | 
```

## Datasets

You can download the official ANI-1 GDB8 dataset [here](https://figshare.com/collections/_/3846712). The GDB10 test
set can be found [here](https://github.com/isayev/ANI1_dataset/blob/master/benchmark/ani1_gdb10_ts.h5).
They should be all put into a directory, eg. ANI_DATA_DIR

Note that ANI-1 and, deep learning in general, is highly dependent on the shape and quality of your dataset.
We've made it very straightforward to include additional datasets beyond GDB8. One needs to either
construct a khan.RawDataset() or implement a class isomorphic to do it. 

``` python
import khan
import tensorflow as tf

# Note that we compact the atomic numbers such that 
# H:1 -> 0
# C:6 -> 1
# N:7 -> 2
# O:8 -> 3

Xs = [
    np.array([
        [0, 1.0, 2.0, 3.0], # H
        [2, 2.0, 1.0, 4.0], # N
        [0, 0.5, 1.2, 2.3], # H
        [1, 0.3, 1.7, 3.2], # C
        [2, 0.6, 1.2, 1.1], # N
        [0, 14.0, 23.0, 15.0], # H
        [0, 2.0, 0.5, 0.3], # H
        [0, 2.3, 0.2, 0.4], # H
    ]),
    np.array([
        [0, 2.3, 0.2, 0.4], # H
        [1, 0.3, 1.7, 3.2], # C
        [2, 0.6, 1.2, 1.1], # N
    ])
]

ys = [1.5, 3.3]

dataset = khan.RawDataset(Xs, ys)

batch_size = 1024

with tf.Session(config=config) as sess:

    # initialize a trainer
    trainer = khan.TrainerMultiGPU(sess, n_gpus=int(args.gpus))

    # train one epoch
    trainer.feed_dataset(dataset, shuffle=True, target_ops=trainer.train_op, batch_size=batch_size)

    # predict
    trainer.predict(dataset)

```

For a more comprehensive example please refer to gdb8.py.


## Current limitations

1. Changing the featurization requires a re-compile (edit gpu_featurizer/parameters.h)
2. 4 atoms types (eg. H C N O)3
3. Closed shell neutral species
4. Single node training

## License

Since we intend to merge this upstream with DeepChem in the near future, this is currently licensed
under the MIT License.

## Copyright

Schrodinger, Inc. 2018