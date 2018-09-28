#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

#include <chrono>

// #include <cmath> needed for std::isfinite when debugging
// #include <math.h>

#include "memcpy.h"
#include "parameters.h"
#include "functor_op.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


template<typename Device>
class AniBase : public OpKernel {

 public:
  explicit AniBase(OpKernelConstruction* context) : OpKernel(context) {

    OP_REQUIRES_OK(context, context->GetAttr("n_types", &n_types_));
    OP_REQUIRES_OK(context, context->GetAttr("R_Rc", &R_Rc_));
    OP_REQUIRES_OK(context, context->GetAttr("R_eta", &R_eta_));
    OP_REQUIRES_OK(context, context->GetAttr("A_Rc", &A_Rc_));
    OP_REQUIRES_OK(context, context->GetAttr("A_eta", &A_eta_));
    OP_REQUIRES_OK(context, context->GetAttr("A_zeta", &A_zeta_));

    // (ytz) there's gotta be a cleaner way to do this.
    std::vector<float> tmp_R_Rs;
    OP_REQUIRES_OK(context, context->GetAttr("R_Rs", &tmp_R_Rs));

    std::vector<float> tmp_A_thetas;
    OP_REQUIRES_OK(context, context->GetAttr("A_thetas", &tmp_A_thetas));

    std::vector<float> tmp_A_Rs;
    OP_REQUIRES_OK(context, context->GetAttr("A_Rs", &tmp_A_Rs));

    context->allocate_persistent(DT_FLOAT, {static_cast<long long>(tmp_R_Rs.size())}, &R_Rs_, nullptr);
    context->allocate_persistent(DT_FLOAT, {static_cast<long long>(tmp_A_thetas.size())}, &A_thetas_, nullptr);
    context->allocate_persistent(DT_FLOAT, {static_cast<long long>(tmp_A_Rs.size())}, &A_Rs_, nullptr);

    device_memcpy<Device>(R_Rs_.AccessTensor(context)->template flat<float>().data(), tmp_R_Rs);
    device_memcpy<Device>(A_thetas_.AccessTensor(context)->template flat<float>().data(), tmp_A_thetas);
    device_memcpy<Device>(A_Rs_.AccessTensor(context)->template flat<float>().data(), tmp_A_Rs);


    // todo: declare const-ness etc.
    params.max_types = n_types_;
    params.R_Rc = R_Rc_;
    params.R_eta = R_eta_;
    params.A_Rc = A_Rc_;
    params.A_eta = A_eta_;
    params.A_zeta = A_zeta_;
    params.Num_R_Rs = R_Rs_.NumElements();
    params.Num_A_thetas = A_thetas_.NumElements();
    params.Num_A_Rs = A_Rs_.NumElements();

    params.R_Rs = R_Rs_.AccessTensor(context)->template flat<float>().data();
    params.A_thetas = A_thetas_.AccessTensor(context)->template flat<float>().data();
    params.A_Rs = A_Rs_.AccessTensor(context)->template flat<float>().data();
 }

 protected:

  const AniParams &getAniParams() const {
    return params;
  }


 private:
  int n_types_;
  float R_Rc_;
  float R_eta_;
  float A_Rc_;
  float A_eta_;
  float A_zeta_;
  PersistentTensor R_Rs_;
  PersistentTensor A_thetas_;
  PersistentTensor A_Rs_;

  AniParams params;

};

template<typename Device, typename NumericType>
class AniCombined : public AniBase<Device> {

 public:
  explicit AniCombined(OpKernelConstruction* context) : AniBase<Device>(context) {}

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

    // this-> is required in the case of template inheritance.
    const AniParams& params = this->getAniParams();
    const size_t total_feat_size = params.total_feature_size();

    OP_REQUIRES_OK(context, context->allocate_output(
      "h_feat",
      TensorShape({static_cast<long long>(acs[0]*total_feat_size)}),
      &X_feat_H)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "c_feat",
      TensorShape({static_cast<long long>(acs[1]*total_feat_size)}),
      &X_feat_C)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "n_feat",
      TensorShape({static_cast<long long>(acs[2]*total_feat_size)}),
      &X_feat_N)
    );
    OP_REQUIRES_OK(context, context->allocate_output(
      "o_feat",
      TensorShape({static_cast<long long>(acs[3]*total_feat_size)}),
      &X_feat_O)
    );

    AniFunctor<Device, NumericType>()(
      context->eigen_device<Device>(), 
      input_Xs.flat<NumericType>().data(),
      input_Ys.flat<NumericType>().data(),
      input_Zs.flat<NumericType>().data(),
      input_As.flat<int>().data(),
      input_MOs.flat<int>().data(),
      input_MACs.flat<int>().data(),
      n_mols,
      input_SIs.flat<int>().data(),
      X_feat_H->flat<NumericType>().data(),
      X_feat_C->flat<NumericType>().data(),
      X_feat_N->flat<NumericType>().data(),
      X_feat_O->flat<NumericType>().data(),
      acs,
      params
    );

  }

  private:
    int n_types_;
    float R_Rc_;
    float R_eta_;
    float A_Rc_;
    float A_eta_;
    float A_zeta_;
    PersistentTensor R_Rs_;
    PersistentTensor A_thetas_;
    PersistentTensor A_Rs_;
};

REGISTER_OP("Featurize")
  .Input("xs: FT")
  .Input("ys: FT")
  .Input("zs: FT")
  .Input("as: int32")
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Input("sis: int32") // scatter_idxs
  .Input("acs: int32") // atom counts of size 4 (HOST MEMORY)
  .Output("h_feat: FT")
  .Output("c_feat: FT")
  .Output("n_feat: FT")
  .Output("o_feat: FT")
  // .Attr("feature_size: int = "+std::to_string(TOTAL_FEATURE_SIZE)) // calling code can compute this explicitly
  .Attr("FT: {float32, float64}")
  .Attr("n_types: int")
  .Attr("R_Rs: list(float)")
  .Attr("A_thetas: list(float)")
  .Attr("A_Rs: list(float)")
  .Attr("R_eta: float")
  .Attr("R_Rc: float")
  .Attr("A_Rc: float")
  .Attr("A_eta: float")
  .Attr("A_zeta: float")

  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // the output shapes are determined by the number of elements in acs
    // c->set_output(0, c->input(0));
    // c->set_output(0, c->input(0));
    // c->set_output(0, c->input(0));
    // c->set_output(0, c->input(0));
    return Status::OK();
  });



template<typename Device, typename NumericType>
class AniCombinedGrad : public AniBase<Device> {

 public:
  explicit AniCombinedGrad(OpKernelConstruction* context) : AniBase<Device>(context) {

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

    // this-> is required in the case of template inheritance.
    const AniParams& params = this->getAniParams();

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

    AniGrad<Device, NumericType>()(
      context->eigen_device<Device>(), 
      input_Xs.flat<NumericType>().data(),
      input_Ys.flat<NumericType>().data(),
      input_Zs.flat<NumericType>().data(),
      input_As.flat<int>().data(),
      input_MOs.flat<int>().data(),
      input_MACs.flat<int>().data(),
      n_mols,
      input_SIs.flat<int>().data(),
      input_H_grads.flat<NumericType>().data(),
      input_C_grads.flat<NumericType>().data(),
      input_N_grads.flat<NumericType>().data(),
      input_O_grads.flat<NumericType>().data(),
      X_grads->flat<NumericType>().data(),
      Y_grads->flat<NumericType>().data(),
      Z_grads->flat<NumericType>().data(),
      acs,
      params
    );



    // CPU MEMORY - USE ONLY WHEN DEBUGGING NANS ON CPU
    // CAUSES SEGFAULT ON GPU
    // auto test_ptr_X = X_grads->flat<NumericType>().data();
    // for(size_t i=0; i < total_num_atoms; i++) {
    //   if(!std::isfinite(test_ptr_X[i])) {
    //       throw std::runtime_error("FAILED X");
    //       printf("X d_y_i_NAN\n");
    //   }
    // }

    // auto test_ptr_Y = Y_grads->flat<NumericType>().data();
    // for(size_t i=0; i < total_num_atoms; i++) {
    //   if(!std::isfinite(test_ptr_Y[i])) {
    //       throw std::runtime_error("FAILED Y");
    //       printf("Y d_y_i_NAN\n");
    //   }
    // }

    // auto test_ptr_Z = Z_grads->flat<NumericType>().data();
    // for(size_t i=0; i < total_num_atoms; i++) {
    //   if(!std::isfinite(test_ptr_Z[i])) {
    //       throw std::runtime_error("FAILED Z");
    //       printf("Z d_y_i_NAN\n");

    //   }
    // }


  }
};


REGISTER_OP("FeaturizeGrad")
  .Input("xs: FT")
  .Input("ys: FT")
  .Input("zs: FT")
  .Input("as: int32")
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Input("sis: int32") // scatter_idxs
  .Input("acs: int32") // atom counts of size 4 (HOST MEMORY)
  .Input("h_grads: FT")
  .Input("c_grads: FT")
  .Input("n_grads: FT")
  .Input("o_grads: FT")
  .Output("x_grads: FT")
  .Output("y_grads: FT")
  .Output("z_grads: FT")
  .Attr("FT: {float32, float64}")
  .Attr("n_types: int")
  .Attr("R_Rs: list(float)")
  .Attr("A_thetas: list(float)")
  .Attr("A_Rs: list(float)")
  .Attr("R_eta: float")
  .Attr("R_Rc: float")
  .Attr("A_Rc: float")
  .Attr("A_eta: float")
  .Attr("A_zeta: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    return Status::OK();
  });


template<typename Device, typename NumericType>
class AniCombinedGradInverse : public AniBase<Device> {

 public:
  explicit AniCombinedGradInverse(OpKernelConstruction* context) : AniBase<Device>(context) {

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
    const AniParams& params = this->getAniParams();
    const size_t total_feat_size = params.total_feature_size();

    OP_REQUIRES_OK(context, context->allocate_output(
      "h_grads",
      TensorShape({static_cast<long long>(acs[0]*total_feat_size)}),
      &output_H_grads)
    );

    OP_REQUIRES_OK(context, context->allocate_output(
      "c_grads",
      TensorShape({static_cast<long long>(acs[1]*total_feat_size)}),
      &output_C_grads)
    );

    OP_REQUIRES_OK(context, context->allocate_output(
      "n_grads",
      TensorShape({static_cast<long long>(acs[2]*total_feat_size)}),
      &output_N_grads)
    );

    OP_REQUIRES_OK(context, context->allocate_output(
      "o_grads",
      TensorShape({static_cast<long long>(acs[3]*total_feat_size)}),
      &output_O_grads)
    );

    AniGradInverse<Device, NumericType>()(
      context->eigen_device<Device>(), 
      input_Xs.flat<NumericType>().data(),
      input_Ys.flat<NumericType>().data(),
      input_Zs.flat<NumericType>().data(),
      input_As.flat<int>().data(),
      input_MOs.flat<int>().data(),
      input_MACs.flat<int>().data(),
      n_mols,
      input_SIs.flat<int>().data(),
      input_X_grads.flat<NumericType>().data(),
      input_Y_grads.flat<NumericType>().data(),
      input_Z_grads.flat<NumericType>().data(),
      output_H_grads->flat<NumericType>().data(),
      output_C_grads->flat<NumericType>().data(),
      output_N_grads->flat<NumericType>().data(),
      output_O_grads->flat<NumericType>().data(),
      acs,
      params
    );
  }
};


REGISTER_OP("FeaturizeGradInverse")
  .Input("xs: FT")
  .Input("ys: FT")
  .Input("zs: FT")
  .Input("as: int32")
  .Input("mos: int32") // mol offsets
  .Input("macs: int32") // mol atom counts
  .Input("sis: int32") // scatter_idxs
  .Input("acs: int32") // atom counts of size 4 (HOST MEMORY)
  .Input("x_grads: FT")
  .Input("y_grads: FT")
  .Input("z_grads: FT")
  .Output("h_grads: FT")
  .Output("c_grads: FT")
  .Output("n_grads: FT")
  .Output("o_grads: FT")
  .Attr("FT: {float32, float64}")
  .Attr("n_types: int")
  .Attr("R_Rs: list(float)")
  .Attr("A_thetas: list(float)")
  .Attr("A_Rs: list(float)")
  .Attr("R_eta: float")
  .Attr("R_Rc: float")
  .Attr("A_Rc: float")
  .Attr("A_eta: float")
  .Attr("A_zeta: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    return Status::OK();
  });

#ifdef ANI_GPU
  REGISTER_KERNEL_BUILDER(Name("Featurize").HostMemory("acs").Device(DEVICE_GPU).TypeConstraint<float>("FT"), AniCombined<GPUDevice, float>);
  REGISTER_KERNEL_BUILDER(Name("FeaturizeGrad").HostMemory("acs").Device(DEVICE_GPU).TypeConstraint<float>("FT"), AniCombinedGrad<GPUDevice, float>);
  REGISTER_KERNEL_BUILDER(Name("FeaturizeGradInverse").HostMemory("acs").Device(DEVICE_GPU).TypeConstraint<float>("FT"), AniCombinedGradInverse<GPUDevice, float>);
  // REGISTER_KERNEL_BUILDER(Name("Featurize").HostMemory("acs").Device(DEVICE_GPU).TypeConstraint<double>("FT"), AniCombined<GPUDevice, double>);
#endif
REGISTER_KERNEL_BUILDER(Name("Featurize").HostMemory("acs").Device(DEVICE_CPU).TypeConstraint<double>("FT"), AniCombined<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("Featurize").HostMemory("acs").Device(DEVICE_CPU).TypeConstraint<float>("FT"), AniCombined<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("FeaturizeGrad").HostMemory("acs").Device(DEVICE_CPU).TypeConstraint<double>("FT"), AniCombinedGrad<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("FeaturizeGrad").HostMemory("acs").Device(DEVICE_CPU).TypeConstraint<float>("FT"), AniCombinedGrad<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("FeaturizeGradInverse").HostMemory("acs").Device(DEVICE_CPU).TypeConstraint<double>("FT"), AniCombinedGradInverse<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("FeaturizeGradInverse").HostMemory("acs").Device(DEVICE_CPU).TypeConstraint<float>("FT"), AniCombinedGradInverse<CPUDevice, float>);
