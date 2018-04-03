#ifndef ANI_KERNEL_CUH_
#define ANI_KERNEL_CUH_

__global__ void featurize(
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
    float *X_feat_out_O);

__global__ void initialize(
    float *array,
    float val,
    int n_elems) {

    int elem_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(elem_idx < n_elems) {
        array[elem_idx] = val;
    }

};

__global__ void inverse(
    const int *sort_idxs,
    int *gather_idxs,
    size_t n_elems);

__global__ void scatter(
    const int *sorted_global_idxs,
    const int *sorted_local_idxs,
    int *scatter_idxs,
    size_t n_elems);


#endif