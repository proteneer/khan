// kernel_example.cc

// #define EIGEN_USE_GPU // do *not* remove, this is used by the tf/eigen headers to define GpuDevice types
// #define GOOGLE_CUDA

// #include "ani_op.h"
// compile flags:
// g++ -std=c++11 -shared fast_split_sort_gather.cpp -o ani_sort.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -Ofast -march=native
// g++ -std=c++11 -shared fast_split_sort_gather.cpp -o ani_sort.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -Ofast -march=native -ltensorflow_framework


#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

#include <vector>
#include <algorithm>
#include <numeric>

#include "parameters.h"

#include <cmath>

using namespace tensorflow;

// todo: refactor with functor_op_gpu.cu
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

static inline __device__ float dist_diff(float dx, float dy, float dz) {

    return sqrt(dx*dx+dy*dy+dz*dz);

}

inline __device__ float f_C(float r_ij, float r_c) {
    if (r_ij <= r_c) {
        return 0.5 * cosf((M_PI * r_ij) / r_c) + 0.5;
    } else {
        return 0;
    }
}

// implement AniChargeGPU and AniChargeGPUGrad
REGISTER_OP("AniChargeGPU")
  .Input("xs: float32")
  .Input("ys: float32")
  .Input("zs: float32") // 
  .Input("qs: float32") // partial charges
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Output("charge_energies: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    return Status::OK();
  });

__global__ void charge_nrg(
  const float *xs,
  const float *ys,
  const float *zs,
  const float *qs,

  const int *mol_offsets,
  const int *mol_atom_count,
  const int num_mols,

  float *batch_energies) {

  int mol_idx = blockIdx.x;
  int num_atoms = mol_atom_count[blockIdx.x];
  int block_size = blockDim.x;
  int num_warps = (num_atoms + block_size - 1)/block_size; // how many warps we need to process

  float local_energy = 0; // contribution to the total energy of the molecule
  // by atoms local_atom_idx, 1*block_size + threadIdx.x, 2*block_size + threadIdx.x, etc.
  for(int warp_idx = 0; warp_idx < num_warps; warp_idx++) {

    int local_atom_idx = warp_idx*block_size + threadIdx.x; // local_local_atom_idx

    if (local_atom_idx >= num_atoms) {
        return;
    }

    int g_atom_idx_i = mol_offsets[mol_idx]+local_atom_idx;
    int i = local_atom_idx;

    float i_x = xs[g_atom_idx_i];
    float i_y = ys[g_atom_idx_i];
    float i_z = zs[g_atom_idx_i];
    float i_q = qs[g_atom_idx_i];

    for(size_t j=i+1; j < num_atoms; j++) {

      int g_atom_idx_j = mol_offsets[mol_idx]+j;

      // can replace with shuffle later since other
      // warps have these registers
      float j_x = xs[g_atom_idx_j];
      float j_y = ys[g_atom_idx_j];
      float j_z = zs[g_atom_idx_j];
      float j_q = qs[g_atom_idx_j];

      float dx = i_x - j_x;
      float dy = i_y - j_y;
      float dz = i_z - j_z;

      float r = dist_diff(dx, dy, dz);

      local_energy += CHARGE_CONSTANT*i_q*j_q*(1-f_C(r, R_Rc))/r;
    }
  }
  // reduce with a shuffle later
  // writing to globals but still at least using hw as opposed to replay in sw
  atomicAdd(batch_energies + mol_idx, local_energy);
}

class AniChargeGPU : public OpKernel {

 public:
  explicit AniChargeGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {

    const Tensor& input_xs  = context->input(0);
    const Tensor& input_ys  = context->input(1);
    const Tensor& input_zs  = context->input(2);
    const Tensor& input_qs  = context->input(3);
    const Tensor& input_mos  = context->input(4);
    const Tensor& input_macs  = context->input(5);

    long long num_mols = input_macs.shape().num_elements();

    const float* raw_input_xs = input_xs.flat<float>().data();
    const float* raw_input_ys = input_ys.flat<float>().data();
    const float* raw_input_zs = input_zs.flat<float>().data();
    const float* raw_input_qs = input_qs.flat<float>().data();
    const int* raw_input_mos = input_mos.flat<int>().data();
    const int* raw_input_macs = input_macs.flat<int>().data();

    Tensor *charge_energies;

    OP_REQUIRES_OK(context, context->allocate_output("charge_energies", TensorShape({num_mols}), &charge_energies));

    float* raw_output_charge_energies = charge_energies->flat<float>().data(); // uninitialized but safe

    const Eigen::GpuDevice &device = context->eigen_device<GPUDevice>();

    gpuErrchk(cudaMemsetAsync(raw_output_charge_energies, 0, num_mols*sizeof(float), device.stream()));
    
    if(num_mols > 0) {
      charge_nrg<<<num_mols, 32, 0, device.stream()>>>(
        raw_input_xs,
        raw_input_ys,
        raw_input_zs,
        raw_input_qs,
        raw_input_mos,
        raw_input_macs,
        num_mols,
        raw_output_charge_energies
      );
    }
    gpuErrchk(cudaPeekAtLastError());
  }
};


REGISTER_OP("AniChargeGPUGrad")
  .Input("xs: float32")
  .Input("ys: float32")
  .Input("zs: float32") // 
  .Input("qs: float32") // partial charges
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Input("grads: float32") // input gradients
  .Output("q_grads: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    return Status::OK();
  });



__global__ void charge_grads(
  const float *xs,
  const float *ys,
  const float *zs,
  const float *qs,

  const int *mol_offsets,
  const int *mol_atom_count,
  const int num_mols,
  const float *energy_grads,

  float *batch_grads) {

  int mol_idx = blockIdx.x;

  int num_atoms = mol_atom_count[blockIdx.x];
  int block_size = blockDim.x;
  int num_warps = (num_atoms + block_size - 1)/block_size; // how many warps we need to process

  for(int warp_idx = 0; warp_idx < num_warps; warp_idx++) {
    int local_atom_idx = warp_idx*block_size + threadIdx.x; // local_local_atom_idx

    if (local_atom_idx >= num_atoms) {
        return;
    }

    int g_atom_idx_i = mol_offsets[mol_idx]+local_atom_idx;
    int i = local_atom_idx;

    float local_grad_i = 0;

    float i_x = xs[g_atom_idx_i];
    float i_y = ys[g_atom_idx_i];
    float i_z = zs[g_atom_idx_i];
    float i_q = qs[g_atom_idx_i];

    float nrg_grad = energy_grads[mol_idx];

    for(size_t j=i+1; j < num_atoms; j++) {

      int g_atom_idx_j = mol_offsets[mol_idx]+j;

      // can replace with shuffle later since other
      // warps have these registers
      float j_x = xs[g_atom_idx_j];
      float j_y = ys[g_atom_idx_j];
      float j_z = zs[g_atom_idx_j];
      float j_q = qs[g_atom_idx_j];

      float dx = i_x - j_x;
      float dy = i_y - j_y;
      float dz = i_z - j_z;
      float r = dist_diff(dx, dy, dz);
      local_grad_i += nrg_grad*CHARGE_CONSTANT*j_q*(1-f_C(r, R_Rc))/r;
      // replace with shuffle later
      atomicAdd(batch_grads + g_atom_idx_j, nrg_grad*CHARGE_CONSTANT*i_q*(1-f_C(r, R_Rc))/r);
    }

    // reduce with a shuffle later
    // writing to globals but still at least using hw as opposed to replay in sw
    atomicAdd(batch_grads + g_atom_idx_i, local_grad_i);

  }
}

/*
float charge_grads(const float *xs,  const float *ys, const float *zs, const float *qs, size_t num_atoms, float *q_grads, float grad) {
  for(size_t i=0; i < num_atoms; i++) {
    float q_grad = 0;
    // note: this start at 0 unlike the energy calculation
    for(size_t j=0; j < num_atoms; j++) {
      if(i==j) {
        continue;
      }
      float dx = xs[i] - xs[j];
      float dy = ys[i] - ys[j];
      float dz = zs[i] - zs[j];
      float r = dist_diff(dx, dy, dz);
      q_grad += grad*CHARGE_CONSTANT*qs[j]*(1-f_C(r, R_Rc))/r;
    }
    q_grads[i] = q_grad;
  }
}*/


class AniChargeGPUGrad : public OpKernel {

 public:
  explicit AniChargeGPUGrad(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {

    const Tensor& input_xs  = context->input(0);
    const Tensor& input_ys  = context->input(1);
    const Tensor& input_zs  = context->input(2);
    const Tensor& input_qs  = context->input(3);
    const Tensor& input_mos  = context->input(4);
    const Tensor& input_macs  = context->input(5);
    const Tensor& input_grads  = context->input(6);

    long long num_mols = input_macs.shape().num_elements();
    long long total_num_atoms = input_xs.shape().num_elements();

    const float* raw_input_xs = input_xs.flat<float>().data();
    const float* raw_input_ys = input_ys.flat<float>().data();
    const float* raw_input_zs = input_zs.flat<float>().data();
    const float* raw_input_qs = input_qs.flat<float>().data();
    const int* raw_input_mos = input_mos.flat<int>().data();
    const int* raw_input_macs = input_macs.flat<int>().data();
    const float* raw_input_grads = input_grads.flat<float>().data();

    Tensor *q_grads;

    OP_REQUIRES_OK(context, context->allocate_output("q_grads", TensorShape({total_num_atoms}), &q_grads));

    float* grad_vals = q_grads->flat<float>().data(); // uninitialized but safe


    // OP_REQUIRES_OK(context, context->allocate_output("charge_energies", TensorShape({num_mols}), &charge_energies));

    // float* raw_output_charge_energies = charge_energies->flat<float>().data(); // uninitialized but safe

    const Eigen::GpuDevice &device = context->eigen_device<GPUDevice>();

    gpuErrchk(cudaMemsetAsync(grad_vals, 0, total_num_atoms*sizeof(float), device.stream()));
    
    if(num_mols > 0) {
      charge_grads<<<num_mols, 32, 0, device.stream()>>>(
        raw_input_xs,
        raw_input_ys,
        raw_input_zs,
        raw_input_qs,
        raw_input_mos,
        raw_input_macs,
        num_mols,
        raw_input_grads,
        grad_vals
      );
      gpuErrchk(cudaPeekAtLastError());
    }


  }
};



REGISTER_KERNEL_BUILDER(
  Name("AniCharge").Device(DEVICE_GPU), AniChargeGPU
);


REGISTER_KERNEL_BUILDER(
  Name("AniChargeGrad").Device(DEVICE_GPU), AniChargeGPUGrad
);