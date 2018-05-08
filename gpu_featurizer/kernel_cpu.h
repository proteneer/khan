#ifndef KERNEL_CPU_H_
#define KERNEL_CPU_H_

#include "parameters.h"

void featurize_cpu(
    const float *input_Xs,
    const float *input_Ys,
    const float *input_Zs,
    const int *input_As,
    const int *input_MOs,
    const int *input_MACs,
    const int num_mols,
    const int *input_SIs,
    float *X_feat_Hs,
    float *X_feat_Cs,
    float *X_feat_Ns,
    float *X_feat_Os);

void featurize_grad_cpu(
    const float *input_Xs,
    const float *input_Ys,
    const float *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols, // denotes where the atom is being displaced to
    const int *input_SIs,
    const float *H_grads,
    const float *C_grads,
    const float *N_grads,
    const float *O_grads,
    float *X_grads,
    float *Y_grads,
    float *Z_grads);

#endif