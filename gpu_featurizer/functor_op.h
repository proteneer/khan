#ifndef FUNCTOR_OP_H_
#define FUNCTOR_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "parameters.h"
#include "kernel_cpu.h"

template <typename Device, typename NumericType>
struct AniFunctor {
  void operator()(
    const Device& d,
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
    const int *acs);
};

template <typename Device, typename NumericType>
struct AniGrad {
  void operator()(
    const Device& d,
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
    NumericType *X_grads,
    NumericType *Y_grads,
    NumericType *Z_grads,
    const int *acs);
};

template <typename Device, typename NumericType>
struct AniGradInverse {
  void operator()(
    const Device& d,
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
    const int *acs);
};



#endif