#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

#include "functor_op.h"
#include "parameters.h"
#include "kernel_cpu.h"

using CPUDevice = Eigen::ThreadPoolDevice;


// template specialization
template<typename NumericType>
struct AniFunctor<CPUDevice, NumericType> {
    void operator()(
        const CPUDevice& d,
        const NumericType *Xs,
        const NumericType *Ys,
        const NumericType *Zs,
        const int *atomic_nums,
        const int *mol_offsets,
        const int *mol_atom_count,
        const int num_mols, // actually equal to blockDim.x
        const int *scatter_idxs, // LOCAL WITHIN THE ATOM TYPE
        NumericType *X_feat_out_H,
        NumericType *X_feat_out_C,
        NumericType *X_feat_out_N,
        NumericType *X_feat_out_O,
        const int *acs,
        AniParams params) {

        const size_t total_feat_size = params.total_feature_size();

        memset(X_feat_out_H, 0, acs[0]*total_feat_size*sizeof(NumericType));
        memset(X_feat_out_C, 0, acs[1]*total_feat_size*sizeof(NumericType));
        memset(X_feat_out_N, 0, acs[2]*total_feat_size*sizeof(NumericType));
        memset(X_feat_out_O, 0, acs[3]*total_feat_size*sizeof(NumericType));

        featurize_cpu<NumericType>(
          Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
          X_feat_out_H, X_feat_out_C, X_feat_out_N, X_feat_out_O, params);
    }
};

template struct AniFunctor<CPUDevice, float>;
template struct AniFunctor<CPUDevice, double>;

// template specialization
template<typename NumericType>
struct AniGrad<CPUDevice, NumericType> {
    void operator()(
        const CPUDevice& d,
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

        memset(X_grads_dbl, 0, total_num_atoms*sizeof(NumericType));
        memset(Y_grads_dbl, 0, total_num_atoms*sizeof(NumericType));
        memset(Z_grads_dbl, 0, total_num_atoms*sizeof(NumericType));

        featurize_grad_cpu<NumericType>(
          Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
          input_H_grads, input_C_grads, input_N_grads, input_O_grads,
          X_grads_dbl, Y_grads_dbl, Z_grads_dbl, params);
    }
};

template struct AniGrad<CPUDevice, float>;
template struct AniGrad<CPUDevice, double>;

// template specialization
template <typename NumericType>
struct AniGradInverse<CPUDevice, NumericType> {
  void operator()(
    const CPUDevice& d,
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

    memset(output_H_grads, 0, acs[0]*total_feat_size*sizeof(NumericType));
    memset(output_C_grads, 0, acs[1]*total_feat_size*sizeof(NumericType));
    memset(output_N_grads, 0, acs[2]*total_feat_size*sizeof(NumericType));
    memset(output_O_grads, 0, acs[3]*total_feat_size*sizeof(NumericType));

    featurize_grad_inverse<NumericType>(
      Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
      X_grads, Y_grads, Z_grads,
      output_H_grads, output_C_grads, output_N_grads, output_O_grads, params);
  }
};

// template instantiation
template struct AniGradInverse<CPUDevice, float>;
template struct AniGradInverse<CPUDevice, double>;