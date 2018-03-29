// kernel_example.cc

#define EIGEN_USE_GPU // do *not* remove, this is used by the tf/eigen headers to define GpuDevice types

#include "ani_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
// #include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"

#include "tensorflow/core/util/cuda_kernel_helper.h"


#include <cub/cub.cuh>

#include <chrono>

#include "kernel.cuh"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Ani")
  .Input("xs: float32")
  .Input("ys: float32")
  .Input("zs: float32")
  .Input("as: int32")
  .Input("mos: int32")
  .Input("macs: int32")
  .Output("h_feat: float32")
  .Output("c_feat: float32")
  .Output("n_feat: float32")
  .Output("o_feat: float32")
  .Output("g_idxs: int32")
  // todo: outputlist
  // .Output("x_feat: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}   


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
// template <typename GPUDevice, typename T>
class AniOp : public OpKernel {
  long long counter_;
  long long timer_;

 public:
  explicit AniOp(OpKernelConstruction* context) : 
    OpKernel(context),
    counter_(0),
    timer_(0) {
    // empty constructor
  }

  void Compute(OpKernelContext* context) override {


    // cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    // Grab the input tensor
    const Tensor& input_Xs   = context->input(0);
    const Tensor& input_Ys   = context->input(1);
    const Tensor& input_Zs   = context->input(2);
    const Tensor& input_As   = context->input(3);
    const Tensor& input_MOs  = context->input(4);
    const Tensor& input_MACs = context->input(5);


    long long total_num_atoms = input_Xs.shape().num_elements();
    long long n_mols = input_MOs.shape().num_elements();

    std::vector<int> host_As(total_num_atoms);

    gpuErrchk(cudaMemcpy(&host_As[0], input_As.flat<int>().data(), total_num_atoms*sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> host_A_counts(4, 0);
    std::vector<int> local_idxs(total_num_atoms);

    for(size_t a_idx=0; a_idx < local_idxs.size(); a_idx++) {
      local_idxs[a_idx] = host_A_counts[host_As[a_idx]];
      host_A_counts[host_As[a_idx]] += 1;
    }

    // todo: optimize//initialize on GPU if needed
    std::vector<int> global_idxs(total_num_atoms);
    for(size_t i=0; i < total_num_atoms; i++) {
        global_idxs[i] = i;
    }

    Tensor* X_feat_H = nullptr;
    Tensor* X_feat_C = nullptr;
    Tensor* X_feat_N = nullptr;
    Tensor* X_feat_O = nullptr;
    Tensor* gather_idxs = nullptr;
 
    // std::cout << "START" << std::endl;

    OP_REQUIRES_OK(context, context->allocate_output(
      "h_feat",
      TensorShape({host_A_counts[0]*384}),
      &X_feat_H)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "c_feat",
      TensorShape({host_A_counts[1]*384}),
      &X_feat_C)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "n_feat",
      TensorShape({host_A_counts[2]*384}),
      &X_feat_N)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "o_feat",
      TensorShape({host_A_counts[3]*384}),
      &X_feat_O)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "g_idxs",
      TensorShape({total_num_atoms}),
      &gather_idxs)
    );

    gpuErrchk(cudaMemset(X_feat_H->flat<float>().data(), 0, host_A_counts[0]*384*sizeof(float)));
    gpuErrchk(cudaMemset(X_feat_C->flat<float>().data(), 0, host_A_counts[1]*384*sizeof(float)));
    gpuErrchk(cudaMemset(X_feat_N->flat<float>().data(), 0, host_A_counts[2]*384*sizeof(float)));
    gpuErrchk(cudaMemset(X_feat_O->flat<float>().data(), 0, host_A_counts[3]*384*sizeof(float)));

    Tensor tmp;

    size_t tmp_storage_bytes;

    Tensor sort_idxs_in; // used for both local and global
    OP_REQUIRES_OK(context, context->allocate_temp(
      DT_INT32,
      TensorShape({total_num_atoms}),
      &sort_idxs_in)
    );

    Tensor sort_global_idxs_out;
    OP_REQUIRES_OK(context, context->allocate_temp(
      DT_INT32,
      TensorShape({total_num_atoms}),
      &sort_global_idxs_out)
    );

    Tensor sort_local_idxs_out;
    OP_REQUIRES_OK(context, context->allocate_temp(
      DT_INT32,
      TensorShape({total_num_atoms}),
      &sort_local_idxs_out)
    );

    Tensor keys_out_unused;
    OP_REQUIRES_OK(context, context->allocate_temp(
      DT_INT32,
      TensorShape({total_num_atoms}),
      &keys_out_unused)
    );

    Tensor scatter_idxs;
    OP_REQUIRES_OK(context, context->allocate_temp(
      DT_INT32,
      TensorShape({total_num_atoms}),
      &scatter_idxs)
    );

    gpuErrchk(cudaMemcpy(sort_idxs_in.flat<int>().data(), &global_idxs[0], total_num_atoms*sizeof(int), cudaMemcpyHostToDevice));

    // estimate size requirements
    gpuErrchk(cub::DeviceRadixSort::SortPairs(
      nullptr,
      tmp_storage_bytes,
      input_As.flat<int>().data(),
      keys_out_unused.flat<int>().data(),
      sort_idxs_in.flat<int>().data(),
      sort_global_idxs_out.flat<int>().data(),
      total_num_atoms));

    // std::cout << "asking for: " << tmp_storage_bytes << std::endl;

    context->allocate_temp(
      DT_INT32,
      TensorShape({static_cast<int64>(tmp_storage_bytes)}),
      &tmp);

    // estimate size requirements
    gpuErrchk(cub::DeviceRadixSort::SortPairs(
      tmp.flat<int>().data(),
      tmp_storage_bytes,
      input_As.flat<int>().data(),
      keys_out_unused.flat<int>().data(),
      sort_idxs_in.flat<int>().data(),
      sort_global_idxs_out.flat<int>().data(),
      total_num_atoms));
   
    gpuErrchk(cudaMemcpy(sort_idxs_in.flat<int>().data(), &local_idxs[0], total_num_atoms*sizeof(int), cudaMemcpyHostToDevice));

    /* the next two functions probably extraneous */
    gpuErrchk(cub::DeviceRadixSort::SortPairs(
      nullptr,
      tmp_storage_bytes,
      input_As.flat<int>().data(),
      keys_out_unused.flat<int>().data(),
      sort_idxs_in.flat<int>().data(),
      sort_local_idxs_out.flat<int>().data(),
      total_num_atoms));

    // std::cout << "asking for: " << tmp_storage_bytes << std::endl;

    context->allocate_temp(
      DT_INT32,
      TensorShape({static_cast<int64>(tmp_storage_bytes)}),
      &tmp);

    gpuErrchk(cub::DeviceRadixSort::SortPairs(
      tmp.flat<int>().data(),
      tmp_storage_bytes,
      input_As.flat<int>().data(),
      keys_out_unused.flat<int>().data(),
      sort_idxs_in.flat<int>().data(),
      sort_local_idxs_out.flat<int>().data(),
      total_num_atoms));

    const GPUDevice &d = context->eigen_device<GPUDevice>();

    std::vector<int> host_sort_global_idxs(total_num_atoms);

    cudaMemcpy(&host_sort_global_idxs[0], sort_global_idxs_out.flat<int>().data(), total_num_atoms*sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "host_sort_global_idxs" << std::endl;

    // for(auto v : host_sort_global_idxs) {
    //   std::cout << v <<  " ";
    //   if(v < 0 || v >= total_num_atoms) {
    //     std::cout << "FATAL";
    //     abort();
    //   }
    // }




    scatter<<<n_mols, 32, 0, d.stream()>>>(
      sort_global_idxs_out.flat<int>().data(),
      sort_local_idxs_out.flat<int>().data(),
      scatter_idxs.flat<int>().data(),
      total_num_atoms
    );

    inverse<<<n_mols, 32, 0, d.stream()>>>(
      sort_global_idxs_out.flat<int>().data(),
      gather_idxs->flat<int>().data(),
      total_num_atoms
    );

    gpuErrchk(cudaPeekAtLastError());

    // std::cout << "scatter_idxs" << std::endl;

    // for(auto v : scatter_idxs) {
    //   std::cout << v << " ";
    // }

    auto h_count = host_A_counts[0];
    auto c_count = host_A_counts[1];
    auto n_count = host_A_counts[2];
    auto o_count = host_A_counts[3];

    std::vector<int> host_scatter_idxs(total_num_atoms);

    cudaMemcpy(&host_scatter_idxs[0], scatter_idxs.flat<int>().data(), total_num_atoms*sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "scatter_idxs" << std::endl;

    // for(size_t j=0; j < host_scatter_idxs.size(); j++) {
    //   auto j_anum = host_As[j];

    //   auto scatter_offset = host_scatter_idxs[j];
    //   auto host_A_max = host_A_counts[j_anum];

    //   if(scatter_offset >= host_A_max || scatter_offset < 0) {

    //     std::cout << "BAD: " << j << " " << scatter_offset << " " << host_A_max << std::endl;
    //     abort();
    //   } else {
    //     // std::cout << "OK " << j  << " " << scatter_offset << " " << host_A_max << std::endl;
    //   }
    // }



    std::vector<int> host_mol_offsets(n_mols);

    cudaMemcpy(&host_mol_offsets[0], input_MOs.flat<int>().data(), n_mols*sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "mol_offsets" << std::endl;

    // for(size_t j=0; j < host_mol_offsets.size(); j++) {
    //   std::cout << "mol_offsets: " << host_mol_offsets[j] << std::endl;
    // }



    featurize<<<n_mols, 32, 0, d.stream()>>>(
      input_Xs.flat<float>().data(),
      input_Ys.flat<float>().data(),
      input_Zs.flat<float>().data(),
      input_As.flat<int>().data(),
      input_MOs.flat<int>().data(),
      input_MACs.flat<int>().data(),
      n_mols,
      scatter_idxs.flat<int>().data(),
      X_feat_H->flat<float>().data(),
      X_feat_C->flat<float>().data(),
      X_feat_N->flat<float>().data(),
      X_feat_O->flat<float>().data()
    );

    gpuErrchk(cudaPeekAtLastError());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // cudaDeviceSynchronize();

    timer_ += duration;
    counter_++;




    // cub::DeviceRadixSort::SortPairs(
    //   d_tmp_storage,
    //   d_tmp_storage,
    //   d_As,
    //   d_keys_out,
    //   d_vals_in,
    //   sort_idxs,
    //   sort_num_items);
    //         inverse<<<n_mols, 32>>>(sort_idxs, inv_idxs, sort_num_items);

    // Do the computation.
    // OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                // errors::InvalidArgument("Too many elements in tensor"));
    // ExampleFunctor<GPUDevice, T>()(
    //     context->eigen_device<GPUDevice>(),
    //     static_cast<int>(input_tensor.NumElements()),
    //     input_tensor.flat<T>().data(),
    //     output_tensor->flat<T>().data());
  }
};

// void AniFunctor::operator()(
//     const GPUDevice& d,
//     const float* Xs,
//     const float* Ys,
//     const float* Zs,
//     const int* As,
//     const int* mol_offsets,
//     const int* mol_atom_counts,
//     size_t n_mols,
//     size_t total_num_atoms,

//     float* atom_features,
//     int* sort_idxs) {
  
//     // change to X's in
//     // T* out) {
//     // Launch the cuda kernel.
//     //
//     // See core/util/cuda_kernel_helper.h for example of computing
//     // block count and thread_per_block count.




//         featurize<<<n_mols, 32>>>(
//             d_Xs,
//             d_Ys,
//             d_Zs,
//             d_As,
//             d_MOs,
//             d_MACs,
//             n_mols,
//             inv_idxs,
//             d_X_feat);



//     int block_count = 1024;
//     int thread_per_block = 20;
//     ExampleCudaKernel<T>
//         <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
// }


// Register the CPU kernels.
// #define REGISTER_CPU(T)                                          \
//   REGISTER_KERNEL_BUILDER(                                       \
//       Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
//       AniOp<CPUDevice, T>);
// REGISTER_CPU(float);
// REGISTER_CPU(int32);

// Register the GPU kernels.


// #ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Ani").Device(DEVICE_GPU), AniOp);
// #endif


// #define REGISTER_GPU(T)                                          \
//   /* Declare explicit instantiations in kernel_example.cu.cc. */ \
//   // extern template ExampleFunctor<GPUDevice, float>;              \
//   REGISTER_KERNEL_BUILDER(                                       \
//       Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), AniOp<GPUDevice, T>);
// REGISTER_GPU(float);
// REGISTER_GPU(int32);
// #endif  // GOOGLE_CUDA
