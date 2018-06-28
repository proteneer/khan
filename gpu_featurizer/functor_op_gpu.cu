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
      const int *acs,
      AniParams params) {

      gpuErrchk(cudaMemsetAsync(X_feat_out_H, 0, acs[0]*params.total_feature_size()*sizeof(int), d.stream()));
      gpuErrchk(cudaMemsetAsync(X_feat_out_C, 0, acs[1]*params.total_feature_size()*sizeof(int), d.stream()));
      gpuErrchk(cudaMemsetAsync(X_feat_out_N, 0, acs[2]*params.total_feature_size()*sizeof(int), d.stream()));
      gpuErrchk(cudaMemsetAsync(X_feat_out_O, 0, acs[3]*params.total_feature_size()*sizeof(int), d.stream()));

      if(num_mols > 0) {
        // gpu kernel's can't be launched with a zero blockdim
        featurize_gpu<<<num_mols, 32, 0, d.stream()>>>(
          Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
          X_feat_out_H, X_feat_out_C, X_feat_out_N, X_feat_out_O, params);
        gpuErrchk(cudaPeekAtLastError());
      }
    }
};


// instantiation
template struct AniFunctor<GPUDevice, float>;

template<typename NumericType>
struct AniGrad<GPUDevice, NumericType> {
    void operator()(
        const GPUDevice& d,
        const NumericType *Xs,
        const NumericType *Ys,
        const NumericType *Zs,
        const int *atomic_nums,
        const int *mol_offsets,
        const int *mol_atom_count,
        const int num_mols, // actually equal to blockDim.x
        const int *scatter_idxs, // LOCAL WITHIN THE ATOM TYPE
        const NumericType *input_H_grads,
        const NumericType *input_C_grads,
        const NumericType *input_N_grads,
        const NumericType *input_O_grads,
        NumericType *X_grads_dbl,
        NumericType *Y_grads_dbl,
        NumericType *Z_grads_dbl,
        const int *acs,
        AniParams params) {

        const int total_num_atoms = acs[0] + acs[1] + acs[2] + acs[3];

        gpuErrchk(cudaMemsetAsync(X_grads_dbl, 0, total_num_atoms*sizeof(NumericType), d.stream()));
        gpuErrchk(cudaMemsetAsync(Y_grads_dbl, 0, total_num_atoms*sizeof(NumericType), d.stream()));
        gpuErrchk(cudaMemsetAsync(Z_grads_dbl, 0, total_num_atoms*sizeof(NumericType), d.stream()));

        if(num_mols > 0) {
          featurize_grad_gpu<<<num_mols, 32, 0, d.stream()>>>(
            Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
            input_H_grads, input_C_grads, input_N_grads, input_O_grads,
            X_grads_dbl, Y_grads_dbl, Z_grads_dbl, params);
          gpuErrchk(cudaPeekAtLastError());
        }

    }
};

// instantiation
template struct AniGrad<GPUDevice, float>;

// template specialization
template <typename NumericType>
struct AniGradInverse<GPUDevice, NumericType> {
  void operator()(
    const GPUDevice& d,
    const NumericType *Xs,
    const NumericType *Ys,
    const NumericType *Zs,
    const int *atomic_nums,
    const int *mol_offsets,
    const int *mol_atom_count,
    const int num_mols, // actually equal to blockDim.x
    const int *scatter_idxs, // LOCAL WITHIN THE ATOM TYPE
    const NumericType *X_grads,
    const NumericType *Y_grads,
    const NumericType *Z_grads,
    NumericType *output_H_grads,
    NumericType *output_C_grads,
    NumericType *output_N_grads,
    NumericType *output_O_grads,
    const int *acs,
    AniParams params) {

    const size_t total_feat_size = params.total_feature_size();

    gpuErrchk(cudaMemsetAsync(output_H_grads, 0, acs[0]*total_feat_size*sizeof(NumericType), d.stream()));
    gpuErrchk(cudaMemsetAsync(output_C_grads, 0, acs[1]*total_feat_size*sizeof(NumericType), d.stream()));
    gpuErrchk(cudaMemsetAsync(output_N_grads, 0, acs[2]*total_feat_size*sizeof(NumericType), d.stream()));
    gpuErrchk(cudaMemsetAsync(output_O_grads, 0, acs[3]*total_feat_size*sizeof(NumericType), d.stream()));

    if(num_mols > 0) {
      featurize_grad_inverse_gpu<<<num_mols, 32, 0, d.stream()>>>(
        Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
        X_grads, Y_grads, Z_grads,
        output_H_grads, output_C_grads, output_N_grads, output_O_grads, params);
      gpuErrchk(cudaPeekAtLastError());
    }

  }
};

// instantiation
template struct AniGradInverse<GPUDevice, float>;


#endif 