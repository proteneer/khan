#ifndef KERNEL_CPU_H_
#define KERNEL_CPU_H_

#include "parameters.h"

template<typename NumericType>
void featurize_cpu(
    const NumericType *input_Xs,
    const NumericType *input_Ys,
    const NumericType *input_Zs,
    const int *input_As,
    const int *input_MOs,
    const int *input_MACs,
    const int num_mols,
    const int *input_SIs,
    NumericType *X_feat_Hs,
    NumericType *X_feat_Cs,
    NumericType *X_feat_Ns,
    NumericType *X_feat_Os,
    AniParams params);

template<typename NumericType>
void featurize_grad_cpu(
    const NumericType *input_Xs,
    const NumericType *input_Ys,
    const NumericType *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols, // denotes where the atom is being displaced to
    const int *input_SIs,
    const NumericType *H_grads,
    const NumericType *C_grads,
    const NumericType *N_grads,
    const NumericType *O_grads,
    NumericType *X_grads,
    NumericType *Y_grads,
    NumericType *Z_grads,
    AniParams params);

template<typename NumericType>
void featurize_grad_inverse(
    const NumericType *input_Xs,
    const NumericType *input_Ys,
    const NumericType *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols, // denotes where the atom is being displaced to
    const int *scatter_idxs, // used to retrieve the grad multiplication factor for backprop
    const NumericType *X_grads,
    const NumericType *Y_grads,
    const NumericType *Z_grads, 
    NumericType *H_grads,
    NumericType *C_grads,
    NumericType *N_grads,
    NumericType *O_grads,
    AniParams params);

#endif