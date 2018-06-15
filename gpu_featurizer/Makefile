# Makefile for the custom tensorflow ops required for ANI-1
# By default, this will attempt to build the GPU kernels with NVCC in addition to the CPU kernels.
# If you'd like to build only the cpu kernel, please run: make ani_cpu
# This will generate a cpu only ani_cpu.so file that can be run on machines with or without GPUs.

CUDA_INCLUDE_FLAGS=-I ./symlink -I ./
NVCC=nvcc
TF_CFLAGS=-I/home/yzhao/venv/lib/python3.5/site-packages/tensorflow/include -I/home/yzhao/venv/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -D_GLIBCXX_USE_CXX11_ABI=0
TF_LFLAGS=-L/home/yzhao/venv/lib/python3.5/site-packages/tensorflow -ltensorflow_framework

NVCC_FLAGS=-std=c++11 -arch=sm_61 $(TF_CFLAGS) $(TF_LFLAGS) -Xcompiler -fPIC -O3 -D GOOGLE_CUDA=1 $(CUDA_INCLUDE_FLAGS) --expt-relaxed-constexpr 
CC=g++
# CC_FLAGS=-std=c++11 -O2 -march=native -fPIC $(TF_CFLAGS)
CC_FLAGS=-std=c++11 -O3 -Ofast -march=native -fPIC $(TF_CFLAGS)
CC_LINK_FLAGS=$(TF_LFLAGS)

all: ani

ani: functor_op_cpu.o functor_op_gpu.o kernel.o kernel_cpu.o ani_op_gpu.o ani_sort.o ani_charges.o ani_charges_gpu.o
	$(CC) -shared -o ani_tiny_q.so functor_op_cpu.o functor_op_gpu.o kernel.o kernel_cpu.o ani_charges.o ani_charges_gpu.o ani_op_gpu.o ani_sort.o $(CC_FLAGS) $(CC_LINK_FLAGS)

ani_cpu: functor_op_cpu.o kernel_cpu.o ani_op_cpu.o ani_sort.o ani_charges.o
	$(CC) -shared -o ani_cpu.so functor_op_cpu.o kernel_cpu.o ani_charges.o ani_op_cpu.o ani_sort.o $(CC_FLAGS) $(CC_LINK_FLAGS)

ani_charges_gpu.o: ani_charges_gpu.cu parameters.h
	$(NVCC) ani_charges_gpu.cu -c $(NVCC_FLAGS)

functor_op_gpu.o: functor_op_gpu.cu functor_op.h parameters.h kernel.cuh
	$(NVCC) functor_op_gpu.cu -c $(NVCC_FLAGS)

kernel.o: kernel.cu parameters.h
	$(NVCC) kernel.cu -c $(NVCC_FLAGS)

kernel_cpu.o: kernel_cpu.cpp parameters.h
	$(CC) kernel_cpu.cpp -c $(CC_FLAGS)

functor_op_cpu.o: functor_op_cpu.cpp functor_op.h parameters.h kernel_cpu.h
	$(CC) functor_op_cpu.cpp -c $(CC_FLAGS)

ani_op_gpu.o: ani_op.cpp functor_op.h parameters.h
	$(CC) -o ani_op_gpu.o ani_op.cpp -D ANI_GPU -c $(CC_FLAGS)

ani_op_cpu.o: ani_op.cpp functor_op.h parameters.h
	$(CC) -o ani_op_cpu.o ani_op.cpp -c $(CC_FLAGS)

ani_sort.o: ani_sort.cpp
	$(CC) ani_sort.cpp -c $(CC_FLAGS)

ani_charges.o: ani_charges.cpp parameters.h
	$(CC) ani_charges.cpp -c $(CC_FLAGS)

# numeric_helpers.o: numeric_helpers.cpp
	# $(CC) numeric_helpers.cpp -c $(CC_FLAGS)

clean:
	rm -f *.o

