#!/usr/bin/env bash
export TF_CFLAGS_VAR=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'`
export TF_LFLAGS_VAR=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`
cd gpu_featurizer
make
make ani_cpu