#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

#include "functor_op.h"
#include "parameters.h"
#include "kernel_cpu.h"

using CPUDevice = Eigen::ThreadPoolDevice;

template<>
void AniFunctor<CPUDevice>::operator()(
    const CPUDevice& d,
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

    memset(X_feat_out_H, 0, acs[0]*TOTAL_FEATURE_SIZE*sizeof(int));
    memset(X_feat_out_C, 0, acs[1]*TOTAL_FEATURE_SIZE*sizeof(int));
    memset(X_feat_out_N, 0, acs[2]*TOTAL_FEATURE_SIZE*sizeof(int));
    memset(X_feat_out_O, 0, acs[3]*TOTAL_FEATURE_SIZE*sizeof(int));

    featurize_cpu(
      Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
      X_feat_out_H, X_feat_out_C, X_feat_out_N, X_feat_out_O);

};

template struct AniFunctor<CPUDevice>;

template<>
void AniGrad<CPUDevice>::operator()(
    const CPUDevice& d,
    const float *Xs,
    const float *Ys,
    const float *Zs,
    const int *atomic_nums,
    const int *mol_offsets,
    const int *mol_atom_count,
    const int num_mols, // actually equal to blockDim.x
    const int *scatter_idxs, // LOCAL WITHIN THE ATOM TYPE
    const float *input_H_grads,
    const float *input_C_grads,
    const float *input_N_grads,
    const float *input_O_grads,
    float *X_grads,
    float *Y_grads,
    float *Z_grads,
    const int *acs
    ) {

    const int total_num_atoms = acs[0] + acs[1] + acs[2] + acs[3];

    memset(X_grads, 0, total_num_atoms*sizeof(float));
    memset(Y_grads, 0, total_num_atoms*sizeof(float));
    memset(Z_grads, 0, total_num_atoms*sizeof(float));

    featurize_grad_cpu(
      Xs, Ys, Zs, atomic_nums, mol_offsets, mol_atom_count, num_mols, scatter_idxs,
      input_H_grads, input_C_grads, input_N_grads, input_O_grads,
      X_grads, Y_grads, Z_grads
    );

};

template struct AniGrad<CPUDevice>;
