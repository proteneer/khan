// kernel_example.cc

// #define EIGEN_USE_GPU // do *not* remove, this is used by the tf/eigen headers to define GpuDevice types
// #define GOOGLE_CUDA

// #include "ani_op.h"
// compile flags:
// g++ -std=c++11 -shared fast_split_sort_gather.cpp -o ani_sort.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -Ofast -march=native
// g++ -std=c++11 -shared fast_split_sort_gather.cpp -o ani_sort.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -Ofast -march=native -ltensorflow_framework


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

using CPUDevice = Eigen::ThreadPoolDevice;

const long long num_atom_types = 4;

static inline float dist_diff(float dx, float dy, float dz) {

    return sqrt(dx*dx+dy*dy+dz*dz);

}

const float K_CONST = 0.529176917;

inline float f_C(float r_ij, float r_c) {
    if (r_ij <= r_c) {
        return 0.5 * cosf((M_PI * r_ij) / r_c) + 0.5;
    } else {
        return 0;
    }
}

// implement AniCharge and AniChargeGrad
REGISTER_OP("AniCharge")
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

float charge_energy(const float *xs,  const float *ys, const float *zs, const float *qs, size_t num_atoms) {
  float energy = 0;
  for(size_t i=0; i < num_atoms; i++) {
    for(size_t j=i+1; j < num_atoms; j++) {
      float dx = xs[i] - xs[j];
      float dy = ys[i] - ys[j];
      float dz = zs[i] - zs[j];
      float r = dist_diff(dx, dy, dz);
      if(std::isnan(r)) {
        std::cout << "OMFG NAN" << dx << " " << dy << " " << dz << std::endl;
      }
      energy += K_CONST*qs[i]*qs[j]*(1-f_C(r, R_Rc))/r;
    }
  }
  // std::cout << "energy:" << energy << std::endl;
  return energy;
}

class AniCharge : public OpKernel {

 public:
  explicit AniCharge(OpKernelConstruction* context) : OpKernel(context) {

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

    // (ytz): probably parallelizable at some point
    for(size_t m_idx=0; m_idx < num_mols; m_idx++) {

      size_t offset = raw_input_mos[m_idx];
      size_t num_atoms = raw_input_macs[m_idx];

      raw_output_charge_energies[m_idx] = charge_energy(
        raw_input_xs+offset,
        raw_input_ys+offset,
        raw_input_zs+offset,
        raw_input_qs+offset,
        num_atoms);
    }

  }
};


REGISTER_OP("AniChargeGrad")
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
      q_grad += grad*K_CONST*qs[j]*(1-f_C(r, R_Rc))/r;
    }
    q_grads[i] = q_grad;
  }
}


class AniChargeGrad : public OpKernel {

 public:
  explicit AniChargeGrad(OpKernelConstruction* context) : OpKernel(context) {

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

    // (ytz): probably parallelizable at some point
    for(size_t m_idx=0; m_idx < num_mols; m_idx++) {

      size_t offset = raw_input_mos[m_idx];
      size_t num_atoms = raw_input_macs[m_idx];

      charge_grads(
        raw_input_xs+offset,
        raw_input_ys+offset,
        raw_input_zs+offset,
        raw_input_qs+offset,
        num_atoms,
        grad_vals+offset,
        raw_input_grads[m_idx]
      );
    }

  }
};


REGISTER_KERNEL_BUILDER(
  Name("AniCharge").Device(DEVICE_CPU), AniCharge
);

REGISTER_KERNEL_BUILDER(
  Name("AniChargeGrad").Device(DEVICE_CPU), AniChargeGrad
);