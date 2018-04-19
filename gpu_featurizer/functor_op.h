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

#endif