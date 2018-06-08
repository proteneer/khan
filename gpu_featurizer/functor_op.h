#ifndef FUNCTOR_OP_H_
#define FUNCTOR_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

template <typename Device>
struct AniFunctor {
  void operator()(
    const Device& d,
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
    const int *acs);
};

template <typename Device>
struct AniGrad {
  void operator()(
    const Device& d,
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
    const int *acs);
};

template <typename Device>
struct AniGradInverse {
  void operator()(
    const Device& d,
    const float *Xs,
    const float *Ys,
    const float *Zs,
    const int *atomic_nums,
    const int *mol_offsets,
    const int *mol_atom_count,
    const int num_mols, // actually equal to blockDim.x
    const int *scatter_idxs, // LOCAL WITHIN THE ATOM TYPE
    const float *X_grads,
    const float *Y_grads,
    const float *Z_grads,
    float *output_H_grads,
    float *output_C_grads,
    float *output_N_grads,
    float *output_O_grads,
    const int *acs);
};



#endif