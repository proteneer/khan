// compiler flags:
// compile cpu: g++ -fPIC -O3 -Ofast -march=native -c --std=c++11 kernel_cpu.cpp

// nvcc -std=c++11 -arch=sm_61 -shared ani_op.cc.cu kernel.cu -o ani.so ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I ~/Code/cub-1.8.0/ -Xcompiler -fPIC -O3 -D GOOGLE_CUDA=1 -I ~/cuda_cluster_pkgs_rhel6/usr/local/ --expt-relaxed-constexpr -ltensorflow_framework

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

#include <chrono>

#include "parameters.h"
#include "functor_op.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template<typename Device>
class AniCombined : public OpKernel {

 public:
  explicit AniCombined(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {


    auto start = std::chrono::high_resolution_clock::now();

    // Grab the input tensors
    const Tensor& input_Xs   = context->input(0);
    const Tensor& input_Ys   = context->input(1);
    const Tensor& input_Zs   = context->input(2);
    const Tensor& input_As   = context->input(3);
    const Tensor& input_MOs  = context->input(4);
    const Tensor& input_MACs = context->input(5);
    const Tensor& input_SIs  = context->input(6);
    const Tensor& input_ACs  = context->input(7); // HOST

    long long total_num_atoms = input_Xs.shape().num_elements();
    long long n_mols = input_MOs.shape().num_elements();

    Tensor* X_feat_H = nullptr;
    Tensor* X_feat_C = nullptr;
    Tensor* X_feat_N = nullptr;
    Tensor* X_feat_O = nullptr;
 
    const int *acs = input_ACs.flat<int>().data(); // safe since we declare this to be on the host.

    OP_REQUIRES_OK(context, context->allocate_output(
      "h_feat",
      TensorShape({acs[0]*TOTAL_FEATURE_SIZE}),
      &X_feat_H)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "c_feat",
      TensorShape({acs[1]*TOTAL_FEATURE_SIZE}),
      &X_feat_C)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "n_feat",
      TensorShape({acs[2]*TOTAL_FEATURE_SIZE}),
      &X_feat_N)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "o_feat",
      TensorShape({acs[3]*TOTAL_FEATURE_SIZE}),
      &X_feat_O)
    );

    AniFunctor<Device>()(
      context->eigen_device<Device>(), 
      input_Xs.flat<float>().data(),
      input_Ys.flat<float>().data(),
      input_Zs.flat<float>().data(),
      input_As.flat<int>().data(),
      input_MOs.flat<int>().data(),
      input_MACs.flat<int>().data(),
      n_mols,
      input_SIs.flat<int>().data(),
      X_feat_H->flat<float>().data(),
      X_feat_C->flat<float>().data(),
      X_feat_N->flat<float>().data(),
      X_feat_O->flat<float>().data(),
      acs
    );

  }
};

REGISTER_OP("Featurize")
  .Input("xs: float32")
  .Input("ys: float32")
  .Input("zs: float32")
  .Input("as: int32")
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Input("sis: int32") // scatter_idxs
  .Input("acs: int32") // atom counts of size 4 (HOST MEMORY)
  .Output("h_feat: float32")
  .Output("c_feat: float32")
  .Output("n_feat: float32")
  .Output("o_feat: float32")
  .Attr("feature_size: int = "+std::to_string(TOTAL_FEATURE_SIZE))
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // the output shapes are determined by the number of elements in acs
    // c->set_output(0, c->input(0));
    // c->set_output(0, c->input(0));
    // c->set_output(0, c->input(0));
    // c->set_output(0, c->input(0));
    return Status::OK();
  });



template<typename Device>
class AniCombinedGrad : public OpKernel {

 public:
  explicit AniCombinedGrad(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {


    auto start = std::chrono::high_resolution_clock::now();

    // Grab the input tensors
    const Tensor& input_Xs   = context->input(0);
    const Tensor& input_Ys   = context->input(1);
    const Tensor& input_Zs   = context->input(2);
    const Tensor& input_As   = context->input(3);
    const Tensor& input_MOs  = context->input(4);
    const Tensor& input_MACs = context->input(5);
    const Tensor& input_SIs  = context->input(6);
    const Tensor& input_ACs  = context->input(7); // HOST

    const Tensor& input_H_grads = context->input(8);
    const Tensor& input_C_grads = context->input(9);
    const Tensor& input_N_grads = context->input(10);
    const Tensor& input_O_grads = context->input(11);

    long long total_num_atoms = input_Xs.shape().num_elements();
    long long n_mols = input_MOs.shape().num_elements();

    Tensor* X_grads = nullptr;
    Tensor* Y_grads = nullptr;
    Tensor* Z_grads = nullptr;
 
    const int *acs = input_ACs.flat<int>().data(); // safe since we declare this to be on the host.

    OP_REQUIRES_OK(context, context->allocate_output(
      "x_grads",
      TensorShape({total_num_atoms}),
      &X_grads)
    );

    OP_REQUIRES_OK(context, context->allocate_output(
      "y_grads",
      TensorShape({total_num_atoms}),
      &Y_grads)
    );

    OP_REQUIRES_OK(context, context->allocate_output(
      "z_grads",
      TensorShape({total_num_atoms}),
      &Z_grads)
    );

    AniGrad<Device>()(
      context->eigen_device<Device>(), 
      input_Xs.flat<float>().data(),
      input_Ys.flat<float>().data(),
      input_Zs.flat<float>().data(),
      input_As.flat<int>().data(),
      input_MOs.flat<int>().data(),
      input_MACs.flat<int>().data(),
      n_mols,
      input_SIs.flat<int>().data(),
      input_H_grads.flat<float>().data(),
      input_C_grads.flat<float>().data(),
      input_N_grads.flat<float>().data(),
      input_O_grads.flat<float>().data(),
      X_grads->flat<float>().data(),
      Y_grads->flat<float>().data(),
      Z_grads->flat<float>().data(),
      acs
    );

  }
};


REGISTER_OP("FeaturizeGrad")
  .Input("xs: float32")
  .Input("ys: float32")
  .Input("zs: float32")
  .Input("as: int32")
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Input("sis: int32") // scatter_idxs
  .Input("acs: int32") // atom counts of size 4 (HOST MEMORY)
  .Input("h_grads: float32")
  .Input("c_grads: float32")
  .Input("n_grads: float32")
  .Input("o_grads: float32")
  .Output("x_grads: float32")
  .Output("y_grads: float32")
  .Output("z_grads: float32")
  .Attr("feature_size: int = "+std::to_string(TOTAL_FEATURE_SIZE))
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    return Status::OK();
  });


template<typename Device>
class AniCombinedGradInverse : public OpKernel {

 public:
  explicit AniCombinedGradInverse(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {


    auto start = std::chrono::high_resolution_clock::now();

    // Grab the input tensors
    const Tensor& input_Xs   = context->input(0);
    const Tensor& input_Ys   = context->input(1);
    const Tensor& input_Zs   = context->input(2);
    const Tensor& input_As   = context->input(3);
    const Tensor& input_MOs  = context->input(4);
    const Tensor& input_MACs = context->input(5);
    const Tensor& input_SIs  = context->input(6);
    const Tensor& input_ACs  = context->input(7); // HOST


    const Tensor& input_X_grads = context->input(8);
    const Tensor& input_Y_grads = context->input(9);
    const Tensor& input_Z_grads = context->input(10);

    long long total_num_atoms = input_Xs.shape().num_elements();
    long long n_mols = input_MOs.shape().num_elements();

    Tensor* output_H_grads = nullptr;
    Tensor* output_C_grads = nullptr;
    Tensor* output_N_grads = nullptr;
    Tensor* output_O_grads = nullptr;
 
    const int *acs = input_ACs.flat<int>().data(); // safe since we declare this to be on the host.

    OP_REQUIRES_OK(context, context->allocate_output(
      "h_grads",
      TensorShape({acs[0]*TOTAL_FEATURE_SIZE}),
      &output_H_grads)
    );

    OP_REQUIRES_OK(context, context->allocate_output(
      "c_grads",
      TensorShape({acs[1]*TOTAL_FEATURE_SIZE}),
      &output_C_grads)
    );

    OP_REQUIRES_OK(context, context->allocate_output(
      "n_grads",
      TensorShape({acs[2]*TOTAL_FEATURE_SIZE}),
      &output_N_grads)
    );

    OP_REQUIRES_OK(context, context->allocate_output(
      "o_grads",
      TensorShape({acs[3]*TOTAL_FEATURE_SIZE}),
      &output_O_grads)
    );

    AniGradInverse<Device>()(
      context->eigen_device<Device>(), 
      input_Xs.flat<float>().data(),
      input_Ys.flat<float>().data(),
      input_Zs.flat<float>().data(),
      input_As.flat<int>().data(),
      input_MOs.flat<int>().data(),
      input_MACs.flat<int>().data(),
      n_mols,
      input_SIs.flat<int>().data(),
      input_X_grads.flat<float>().data(),
      input_Y_grads.flat<float>().data(),
      input_Z_grads.flat<float>().data(),
      output_H_grads->flat<float>().data(),
      output_C_grads->flat<float>().data(),
      output_N_grads->flat<float>().data(),
      output_O_grads->flat<float>().data(),
      acs
    );
  }
};


REGISTER_OP("FeaturizeGradInverse")
  .Input("xs: float32")
  .Input("ys: float32")
  .Input("zs: float32")
  .Input("as: int32")
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Input("sis: int32") // scatter_idxs
  .Input("acs: int32") // atom counts of size 4 (HOST MEMORY)
  .Input("x_grads: float32")
  .Input("y_grads: float32")
  .Input("z_grads: float32")
  .Output("h_grads: float32")
  .Output("c_grads: float32")
  .Output("n_grads: float32")
  .Output("o_grads: float32")
  .Attr("feature_size: int = "+std::to_string(TOTAL_FEATURE_SIZE))
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    return Status::OK();
  });

#ifdef ANI_GPU
  REGISTER_KERNEL_BUILDER(Name("Featurize").HostMemory("acs").Device(DEVICE_GPU), AniCombined<GPUDevice>);
#endif
REGISTER_KERNEL_BUILDER(Name("Featurize").HostMemory("acs").Device(DEVICE_CPU), AniCombined<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("FeaturizeGrad").HostMemory("acs").Device(DEVICE_CPU), AniCombinedGrad<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("FeaturizeGradInverse").HostMemory("acs").Device(DEVICE_CPU), AniCombinedGradInverse<CPUDevice>);
