CUDA_INCLUDE_FLAGS=-I /usr/local/
NVCC=nvcc
TF_CFLAGS=-I/home/yutong/venv/lib/python3.5/site-packages/tensorflow/include -I/home/yutong/venv/lib/python3.5/site-packages/tensorflow/include/external/nsync/public
TF_LFLAGS=-L/home/yutong/venv/lib/python3.5/site-packages/tensorflow

NVCC_FLAGS=-std=c++11 -arch=sm_61 $(TF_CFLAGS) $(TF_LFLAGS) -Xcompiler -fPIC -O3 -D GOOGLE_CUDA=1 $(CUDA_INCLUDE_FLAGS) --expt-relaxed-constexpr
CC=g++
CC_FLAGS=-std=c++11 -O3 -Ofast -march=native -fPIC $(TF_CFLAGS) $(TF_LFLAGS)

all: ani

ani: functor_op_cpu.o functor_op_gpu.o kernel.o kernel_cpu.o ani_op.o ani_sort.o
	$(CC) -shared -o ani.so functor_op_cpu.o functor_op_gpu.o kernel.o kernel_cpu.o ani_op.o ani_sort.o $(CC_FLAGS)

functor_op_gpu.o: functor_op_gpu.cu
	$(NVCC) functor_op_gpu.cu -c $(NVCC_FLAGS)

kernel.o: kernel.cu
	$(NVCC) kernel.cu -c $(NVCC_FLAGS)

kernel_cpu.o: kernel_cpu.cpp
	$(CC) kernel_cpu.cpp -c $(CC_FLAGS)

functor_op_cpu.o: functor_op_cpu.cpp
	$(CC) functor_op_cpu.cpp -c $(CC_FLAGS)

ani_op.o: ani_op.cpp
	$(CC) ani_op.cpp -c $(CC_FLAGS)

ani_sort.o: ani_sort.cpp
	$(CC) ani_sort.cpp -c $(CC_FLAGS)

clean:
	rm -f *.o *.so