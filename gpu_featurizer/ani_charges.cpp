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

// #include "parameters.h"
// #include "numeric_helpers.h"

const float R_Rc = 4.6;

const float CHARGE_CONSTANT = 0.529176917; // Coulomb's constant in Hartree, atoms, and atomic charges


#include <cmath>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

const long long num_atom_types = 4;

static inline double dist_diff(double dx, double dy, double dz) {
    return sqrt(dx*dx+dy*dy+dz*dz);
}

static inline double f_C(double r_ij, double r_c) {
    if (r_ij <= r_c) {
        return 0.5 * cos((M_PI * r_ij) / r_c) + 0.5;
    } else {
        return 0;
    }
}

// implement AniCharge and AniChargeGrad
REGISTER_OP("AniCharge")
  .Input("xs: FT")
  .Input("ys: FT")
  .Input("zs: FT") // 
  .Input("qs: FT") // partial charges
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Output("charge_energies: FT")
  .Attr("FT: {float32, float64}")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    return Status::OK();
  });

template<typename NumericType>
NumericType charge_energy(const NumericType *xs,  const NumericType *ys, const NumericType *zs, const NumericType *qs, size_t num_atoms) {
  NumericType energy = 0;
  for(size_t i=0; i < num_atoms; i++) {
    for(size_t j=i+1; j < num_atoms; j++) {
      NumericType dx = xs[i] - xs[j];
      NumericType dy = ys[i] - ys[j];
      NumericType dz = zs[i] - zs[j];
      NumericType r = dist_diff(dx, dy, dz);
      energy += CHARGE_CONSTANT*qs[i]*qs[j]*(1-f_C(r, R_Rc))/r;
    }
  }
  return energy;
}

template<typename NumericType>
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

    const NumericType* raw_input_xs = input_xs.flat<NumericType>().data();
    const NumericType* raw_input_ys = input_ys.flat<NumericType>().data();
    const NumericType* raw_input_zs = input_zs.flat<NumericType>().data();
    const NumericType* raw_input_qs = input_qs.flat<NumericType>().data();
    const int* raw_input_mos = input_mos.flat<int>().data();
    const int* raw_input_macs = input_macs.flat<int>().data();

    Tensor *charge_energies;

    OP_REQUIRES_OK(context, context->allocate_output("charge_energies", TensorShape({num_mols}), &charge_energies));

    NumericType* raw_output_charge_energies = charge_energies->flat<NumericType>().data(); // uninitialized but safe

    // (ytz): probably parallelizable at some point
    for(size_t m_idx=0; m_idx < num_mols; m_idx++) {

      size_t offset = raw_input_mos[m_idx];
      size_t num_atoms = raw_input_macs[m_idx];

      raw_output_charge_energies[m_idx] = charge_energy<NumericType>(
        raw_input_xs+offset,
        raw_input_ys+offset,
        raw_input_zs+offset,
        raw_input_qs+offset,
        num_atoms);
    }

  }
};


REGISTER_OP("AniChargeGrad")
  .Input("xs: FT")
  .Input("ys: FT")
  .Input("zs: FT") // 
  .Input("qs: FT") // partial charges
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Input("grads: FT") // input gradients
  .Output("q_grads: FT")
  .Attr("FT: {float32, float64}")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    return Status::OK();
  });


template<typename NumericType>
NumericType charge_grads(const NumericType *xs,  const NumericType *ys, const NumericType *zs, const NumericType *qs, size_t num_atoms, NumericType *q_grads, NumericType nrg_grad) {
  for(size_t i=0; i < num_atoms; i++) {
    NumericType q_grad_i = 0;
    // note: this start at 0 unlike the energy calculation
    for(size_t j=i+1; j < num_atoms; j++) {
      if(i==j) {
        continue;
      }
      NumericType dx = xs[i] - xs[j];
      NumericType dy = ys[i] - ys[j];
      NumericType dz = zs[i] - zs[j];
      
      NumericType r = dist_diff(dx, dy, dz);
      // accum i locally
      q_grad_i += nrg_grad*CHARGE_CONSTANT*qs[j]*(1-f_C(r, R_Rc))/r;

      // accum j globally
      q_grads[j] += nrg_grad*CHARGE_CONSTANT*qs[i]*(1-f_C(r, R_Rc))/r;
    }
    q_grads[i] += q_grad_i;
  }
}

template<typename NumericType>
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

    const NumericType* raw_input_xs = input_xs.flat<NumericType>().data();
    const NumericType* raw_input_ys = input_ys.flat<NumericType>().data();
    const NumericType* raw_input_zs = input_zs.flat<NumericType>().data();
    const NumericType* raw_input_qs = input_qs.flat<NumericType>().data();
    const int* raw_input_mos = input_mos.flat<int>().data();
    const int* raw_input_macs = input_macs.flat<int>().data();
    const NumericType* raw_input_grads = input_grads.flat<NumericType>().data();

    Tensor *q_grads;

    OP_REQUIRES_OK(context, context->allocate_output("q_grads", TensorShape({total_num_atoms}), &q_grads));

    NumericType* grad_vals = q_grads->flat<NumericType>().data(); // uninitialized but safe

    memset(grad_vals, 0, total_num_atoms*sizeof(NumericType));

    // (ytz): probably parallelizable at some point
    for(size_t m_idx=0; m_idx < num_mols; m_idx++) {

      size_t offset = raw_input_mos[m_idx];
      size_t num_atoms = raw_input_macs[m_idx];

      charge_grads<NumericType>(
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


REGISTER_KERNEL_BUILDER(Name("AniCharge").Device(DEVICE_CPU).TypeConstraint<float>("FT"), AniCharge<float>);
REGISTER_KERNEL_BUILDER(Name("AniCharge").Device(DEVICE_CPU).TypeConstraint<double>("FT"), AniCharge<double>);

REGISTER_KERNEL_BUILDER(Name("AniChargeGrad").Device(DEVICE_CPU).TypeConstraint<float>("FT"), AniChargeGrad<float>);
REGISTER_KERNEL_BUILDER(Name("AniChargeGrad").Device(DEVICE_CPU).TypeConstraint<double>("FT"), AniChargeGrad<double>);