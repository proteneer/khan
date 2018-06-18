// nvcc -std=c++11 -arch=sm_61 -shared ani_op.cc.cu kernel_cpu.o kernel.cu -o ani.so ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I ~/Code/cub-1.8.0/ -Xcompiler -fPIC -O3 -D GOOGLE_CUDA=1 -I /usr/local/ --expt-relaxed-constexpr -ltensorflow_framework
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#include "functor_op.h"
#include "parameters.h"
#include "kernel.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using GPUDevice = Eigen::GpuDevice;
    
// template<>
// void AniFunctor<GPUDevice>::operator()(
//     const GPUDevice& d,
//     const float *Xs,
//     const float *Ys,
//     const float *Zs,
//     const int *atomic_nums,
//     const int *mol_offsets,
//     const int *mol_atom_count,
//     const int num_mols, // actually equal to blockDim.x
//     const int *scatter_idxs, // LOCAL WITHIN THE ATOM TYPE
//     float *X_feat_out_H,
//     float *X_feat_out_C,
//     float *X_feat_out_N,
//     float *X_feat_out_O,
//     const int *acs) {

//     gpuErrchk(cudaMemsetAsync(X_feat_out_H, 0, acs[0]*TOTAL_FEATURE_SIZE*sizeof(int), d.stream()));
//     gpuErrchk(cudaMemsetAsync(X_feat_out_C, 0, acs[1]*TOTAL_FEATURE_SIZE*sizeof(int), d.stream()));
//     gpuErrchk(cudaMemsetAsync(X_feat_out_N, 0, acs[2]*TOTAL_FEATURE_SIZE*sizeof(int), d.stream()));
//     gpuErrchk(cudaMemsetAsync(X_feat_out_O, 0, acs[3]*TOTAL_FEATURE_SIZE*sizeof(int), d.stream()));

//     if(num_mols > 0) {
//       // gpu kernel's can't be launched with a zero blockdim
//       featurize<<<num_mols, 32, 0, d.stream()>>>(
//         Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
//         X_feat_out_H, X_feat_out_C, X_feat_out_N, X_feat_out_O);
//       gpuErrchk(cudaPeekAtLastError());
//     }   
// };

// template struct AniFunctor<GPUDevice>;

template<typename NumericType>
struct AniFunctor<GPUDevice, NumericType> {
    void operator()(
      const GPUDevice& d,
      const float *Xs,
      const float *Ys,
      const float *Zs,
      const int *atomic_nums,
      const int *mol_offsets,
      const int *mol_atom_count,
      const int num_mols, // actually equal to blockDim.x
      const int *scatter_idxs, // LOCAL WITHIN THE ATOM TYPE
      float *X_feat_out_H,
      float *X_feat_out_C,
      float *X_feat_out_N,
      float *X_feat_out_O,
      const int *acs) {

      gpuErrchk(cudaMemsetAsync(X_feat_out_H, 0, acs[0]*TOTAL_FEATURE_SIZE*sizeof(int), d.stream()));
      gpuErrchk(cudaMemsetAsync(X_feat_out_C, 0, acs[1]*TOTAL_FEATURE_SIZE*sizeof(int), d.stream()));
      gpuErrchk(cudaMemsetAsync(X_feat_out_N, 0, acs[2]*TOTAL_FEATURE_SIZE*sizeof(int), d.stream()));
      gpuErrchk(cudaMemsetAsync(X_feat_out_O, 0, acs[3]*TOTAL_FEATURE_SIZE*sizeof(int), d.stream()));

      if(num_mols > 0) {
        // gpu kernel's can't be launched with a zero blockdim
        featurize<<<num_mols, 32, 0, d.stream()>>>(
          Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
          X_feat_out_H, X_feat_out_C, X_feat_out_N, X_feat_out_O);
        gpuErrchk(cudaPeekAtLastError());
      }
    }
};


// instantiation
template struct AniFunctor<GPUDevice, float>;

#endif 