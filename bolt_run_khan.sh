#!/bin/bash

#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -P dev_ml
#$ -m n

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/gridengine/lib/lx-amd64:/opt/openmpi/lib:/nfs/utils/stow/Python-3.5.3/lib/:/nfs/utils/stow/cuda-8.0/lib64/:/home/yzhao/libs/cuda/lib64

source /home/yzhao/venv/bin/activate

echo "Diagnostic information..."
nvidia-smi
export CUDA_VISIBLE_DEVICES=`echo $SGE_HGR_gpu | sed -e 's/gpu//g' -e 's/ /,/g'`
echo $CUDA_VISIBLE_DEVICES
echo $SGE_HGR_gpu
# run python, -u means unbuffered stdout
python -u main_2.py --work-dir saved/june12_24 --restart --gpus 0 --cpus 4 --dataset_index 5 --start_batch_size 64 --max_local_epoch_count 50 --test_size 0.2 --train_size 0.8 --ani_lib gpu_featurizer/ani_smaller_q.so

