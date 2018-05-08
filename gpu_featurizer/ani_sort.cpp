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

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

const long long num_atom_types = 4;

REGISTER_OP("AniSort")
  .Input("atomic_numbers: int32")
  .Output("scatter_idxs: int32")
  .Output("gather_idxs: int32")
  .Output("atom_counts: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(0));
    // c->set_output(2, TensorShape({static_cast<size_t>(4)}));
    return Status::OK();
  });


// permute and generate an inverse
std::vector<size_t> perm(const std::vector<size_t> &idx) {

  std::vector<size_t> p(idx.size());

  for(size_t i=0; i < p.size(); i++) {
    p[idx[i]] = i;
  }

  return p;

}

std::vector<size_t> sort_indexes(const int *v_start, size_t v_size) {

  std::vector<size_t> idx(v_size);
  std::iota(idx.begin(), idx.end(), 0);

  std::stable_sort(idx.begin(), idx.end(),
       [&v_start](size_t i1, size_t i2) {return *(v_start+i1) < *(v_start+i2);});

  return idx;
}

class AniSort : public OpKernel {

 public:
  explicit AniSort(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {

    const Tensor& input_atomic_numbers  = context->input(0); // HOST
    const int *input_raw_atomic_nums = input_atomic_numbers.flat<int>().data(); // safe since we declare this to be on the host.

    long long total_num_atoms = input_atomic_numbers.shape().num_elements();

    Tensor *scatter_idxs;
    Tensor *gather_idxs;
    Tensor *atom_counts;

    OP_REQUIRES_OK(context, context->allocate_output("scatter_idxs", TensorShape({total_num_atoms}), &scatter_idxs));
    OP_REQUIRES_OK(context, context->allocate_output("gather_idxs", TensorShape({total_num_atoms}), &gather_idxs));
    OP_REQUIRES_OK(context, context->allocate_output("atom_counts", TensorShape({num_atom_types}), &atom_counts));

    int* output_raw_scatter_idxs = scatter_idxs->flat<int>().data();
    int* output_raw_gather_idxs = gather_idxs->flat<int>().data();
    int* output_raw_atom_counts = atom_counts->flat<int>().data();

    output_raw_atom_counts[0] = 0;
    output_raw_atom_counts[1] = 0;
    output_raw_atom_counts[2] = 0;
    output_raw_atom_counts[3] = 0;

    auto local_idxs = std::vector<int>(total_num_atoms, 0);

    for(size_t i=0; i < total_num_atoms; i++) {
      local_idxs[i] = output_raw_atom_counts[input_raw_atomic_nums[i]];
      output_raw_atom_counts[input_raw_atomic_nums[i]] += 1;
    }

    auto sorted_global_idxs = sort_indexes(input_raw_atomic_nums, total_num_atoms);
    auto p = perm(sorted_global_idxs);

    std::vector<int> sorted_local_idxs(total_num_atoms, 0);
    for(size_t i=0; i < total_num_atoms; i++) {
      sorted_local_idxs[p[i]] = local_idxs[i]; 
    }

    for(size_t i=0; i < total_num_atoms; i++) {
      output_raw_scatter_idxs[sorted_global_idxs[i]] = sorted_local_idxs[i];
      output_raw_gather_idxs[sorted_global_idxs[i]] = i;
    }

  }
};



REGISTER_KERNEL_BUILDER(
  Name("AniSort").Device(DEVICE_CPU), AniSort
);