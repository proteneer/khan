# khan: an open-source deep-learning package for learning molecular energies	

## Background:

The original ANI-1 by Smith et al. (doi:10.1039/C6SC05720A) unfortunately did not provide the
training algorithm that was used to generate the released models. As part of Schrodinger's collaboration
with the Pande lab and the DeepChem project, we've taken the initial pure tensorflow implementation
written by Michael Wu (@miaecle) and optimized it significantly to reduce the training time. 

Note: that this code will at some point be merged back upstream with the DeepChem master, once the
build procedures have been streamlined.

## Features

1. A friendly tensorflow environment with custom ops for performance
2. On-the-fly featurization
3. Simple to use python API for training
4. Highly extensible with additional datasets
5. Scalable to multiple gpus on a single node.

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
epochs to converge.

## Installation Procedures

Requirements:

- Linux
- Python 3.x
- CUDA-enabled GPU
- Modern C++11 compatible gcc

Aside from the dependencies in the requirements.txt file, you will need to build two custom ops:

- gpu_featurizer/fast_split_sort_gather.cpp -> mod.ani_sort()
- gpu_featurizer/ani_op.cc.cu -> mod.ani()

These two custom kernels significantly improve the training speed by offloading parts of the
featurization algorithm directly onto the GPU, while retaining a tensorflow-based ecosystem.

The recommended build flags for the fast_split_sort_gather.cpp:

``` bash
g++ -std=c++11 -shared fast_split_sort_gather.cpp -o ani_sort.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
 -O3 -Ofast -march=native -ltensorflow_framework
```

and for ani_op.cc.cu:

``` bash
nvcc -std=c++11 -arch=sm_61 -shared ani_op.cc.cu kernel.cu -o ani.so ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
 -Xcompiler -fPIC -O3 -D GOOGLE_CUDA=1 -I CUDA_INCLUDE_DIR --expt-relaxed-constexpr -ltensorflow_framework
```

It is critically important that the TF_*FLAGS be set to the corresponding virtualenv in which
tensorflow itself is installed in:

``` bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
```

The built .so files should be placed inside the gpu_featurizer dir directly.

## Customization of datasets

Note that ANI-1 and, deep learning in general, is highly dependent on the shape and quality of your dataset.
We've made it very straightforward to include additional datasets beyond GDB8. One needs to either
construct a khan.RawDataset() or implement a class isomorphic to do it. 

``` python
import khan

Xs = [
	np.array([
	    [0, 1.0, 2.0, 3.0], # H
	    [2, 2.0, 1.0, 4.0], # N
	    [0, 0.5, 1.2, 2.3], # H
	    [1, 0.3, 1.7, 3.2], # C
	    [2, 0.6, 1.2, 1.1], # N
	    [0, 14.0, 23.0, 15.0], # H
	    [0, 2.0, 0.5, 0.3], # H
	    [0, 2.3, 0.2, 0.4]
	]),
	np.array([
	    [0, 2.3, 0.2, 0.4], # H
	    [1, 0.3, 1.7, 3.2], # C
	    [2, 0.6, 1.2, 1.1]
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

## Current limitations

1. The maximum number of atoms for training is 32
2. Changing the featurization requires writing some CUDA code.
3. 4 atoms types (eg. H C N O)
4. Closed shell systems
5. Single node training

## License

Since we intend to merge this upstream with DeepChem in the near future, this is currently licensed
under the MIT License.

## Copyright

Schrodinger, Inc. 2018