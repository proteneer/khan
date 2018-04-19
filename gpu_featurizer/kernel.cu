#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>

#include "parameters.h"

/*

YTZ Notes:

This kernel implements the ANI-1 featurization scheme. It can process about 3.6 million samples/minute

*/

inline __device__ float dist_diff(float dx, float dy, float dz) {

    return sqrt(dx*dx+dy*dy+dz*dz);

}

inline __device__ float f_C(float r_ij, float r_c) {
    if (r_ij <= r_c) {
        return 0.5 * cosf((M_PI * r_ij) / r_c) + 0.5;
    } else {
        return 0;
    }
}

// Linearizes the diagonal-inclusive upper right
// triangle of the symmetric ranks 0 and 1 of a rank-4 tensor
// into a linear index 
inline __device__ int linearize(int i, int j, int k, int l) {
    if(j < i) {
        float tmp = i;
        i = j;
        j = tmp;
    }

    const auto N = MAX_ATOM_TYPES;
    const auto K = NUM_A_THETAS;
    const auto L = NUM_A_RS;

    int basis = (N*(N-1)/2 - (N-i) * (N-i-1)/2 +j);
    
    return basis*K*L + k*L + l;
}


__global__ void inverse(
    const int *sort_idxs,
    int *gather_idxs,
    size_t n_elems) {
    int elem_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(elem_idx < n_elems) {
        gather_idxs[sort_idxs[elem_idx]] = elem_idx;        
    }
} 

__global__ void scatter(
    const int *sorted_global_idxs,
    const int *sorted_local_idxs,
    int *scatter_idxs,
    size_t n_elems) {
    int elem_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(elem_idx < n_elems) {
        scatter_idxs[sorted_global_idxs[elem_idx]] = sorted_local_idxs[elem_idx];
    }
}


// Remind yutong to document what these pointers are.
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
    float *X_feat_out_O) {


    int mol_idx = blockIdx.x;
    int local_atom_idx = threadIdx.x; // local_local_atom_idx
    int num_atoms = mol_atom_count[blockIdx.x];

    if (local_atom_idx >= num_atoms) {
        return;
    }

    // todo: cache into shared mem
    // load all the x y z coordinates
    int g_atom_idx_i = mol_offsets[mol_idx]+local_atom_idx;

    int g_atomic_num_i = atomic_nums[g_atom_idx_i];

    float i_x = Xs[g_atom_idx_i];
    float i_y = Ys[g_atom_idx_i];
    float i_z = Zs[g_atom_idx_i];


    // printf("%d %d %d (%f, %f, %f)\n", mol_idx, local_atom_idx, num_atoms, i_x, i_y, i_z);

    // float *X_feat_i = X_feat_out + scatter_idxs[g_atom_idx_i];

    // if(Atom)

    float *X_feat_out_i;
    if(g_atomic_num_i == 0) {
        X_feat_out_i = X_feat_out_H;
    } else if(g_atomic_num_i == 1) {
        X_feat_out_i = X_feat_out_C;
    } else if(g_atomic_num_i == 2) {
        X_feat_out_i = X_feat_out_N;
    } else {
        X_feat_out_i = X_feat_out_O;
    }
  
    float *radial_feature_buffer_i = X_feat_out_i + scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + 0;
    float *angular_feature_buffer_i = X_feat_out_i + scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + RADIAL_FEATURE_SIZE;

    for(int j=0; j < num_atoms; j++) {

        int g_atom_idx_j = mol_offsets[mol_idx]+j;
        int g_atomic_num_j = atomic_nums[g_atom_idx_j];


        float j_x = Xs[g_atom_idx_j];
        float j_y = Ys[g_atom_idx_j];
        float j_z = Zs[g_atom_idx_j];

        float d_ij_x = i_x - j_x;
        float d_ij_y = i_y - j_y;
        float d_ij_z = i_z - j_z;

        // printf("(%f, %f, %f)\n", d_ij_x, d_ij_y, d_ij_z);

        float r_ij = dist_diff(d_ij_x, d_ij_y, d_ij_z);

        // float *X_feat_j = X_feat_out + scatter_idxs[g_atom_idx_j];

        float *X_feat_out_j;
        if(g_atomic_num_j == 0) {
            X_feat_out_j = X_feat_out_H;
        } else if(g_atomic_num_j == 1) {
            X_feat_out_j = X_feat_out_C;
        } else if(g_atomic_num_j == 2) {
            X_feat_out_j = X_feat_out_N;
        } else {
            X_feat_out_j = X_feat_out_O;
        }

        float *radial_feature_buffer_j = X_feat_out_j + scatter_idxs[g_atom_idx_j]*TOTAL_FEATURE_SIZE + 0;
        // float *radial_feature_buffer_j = X_feat_j + g_atom_idx_j*TOTAL_FEATURE_SIZE + 0;
        // float *angular_feature_buffer_j = X_feat_out + g_atom_idx_j*TOTAL_FEATURE_SIZE + RADIAL_FEATURE_SIZE;


        // if(g_atom_idx_i == 0) {
            // printf("gpu j %d %f\n", j, r_ij);
            // printf("summand, offset, %f, %d\n", summand, scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + atomic_nums[g_atom_idx_j] * NUM_R_Rs + r_idx);                    
            // printf("summand, offset, %f, %d\n", summand, scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + atomic_nums[g_atom_idx_j] * NUM_R_Rs + r_idx);                    
        // }

        // radial features
        if(r_ij < R_Rc && local_atom_idx < j) {
            for(int r_idx = 0; r_idx < NUM_R_Rs; r_idx++) {
                float summand = expf(-R_eta * powf(r_ij - R_Rs[r_idx], 2.0)) * f_C(r_ij, R_Rc);

                // exploit symmetry of the atomic adds
                auto res1 = atomicAdd(radial_feature_buffer_i + atomic_nums[g_atom_idx_j] * NUM_R_Rs + r_idx, summand);
                auto res2 = atomicAdd(radial_feature_buffer_j + atomic_nums[g_atom_idx_i] * NUM_R_Rs + r_idx, summand);
            
                if(isnan(res1) || isinf(res1)) {
                    printf("WTF RADIAL RES1 NAN/INF, offset, %f, %f\n", res1, summand);
                    // : %d, %d, %d, r_ij, r_ik, %f, %f, top %f, bottom %f, i_coords:(%f, %f, %f), j_coords(%f, %f, %f), k_coords(%f, %f, %f)\n",
                        // g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, r_ij, r_ik, d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z, r_ij * r_ik, i_x, i_y, i_z, j_x, j_y, j_z, k_x, k_y, k_z);
                }

                if(isnan(res2) || isinf(res2)) {
                    printf("WTF RADIAL RES2 NAN/INF, offset, %f, %f\n", res2, summand);
                    // : %d, %d, %d, r_ij, r_ik, %f, %f, top %f, bottom %f, i_coords:(%f, %f, %f), j_coords(%f, %f, %f), k_coords(%f, %f, %f)\n",
                        // g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, r_ij, r_ik, d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z, r_ij * r_ik, i_x, i_y, i_z, j_x, j_y, j_z, k_x, k_y, k_z);
                }

            }
        }

        float A_f_C_ij = f_C(r_ij, A_Rc);


        if(r_ij < A_Rc) {

            for(size_t k=j+1; k < num_atoms; k++) {

                if(local_atom_idx == j || local_atom_idx == k || j == k) {
                    continue;
                }

                // const int an_i = atomic_nums[local_atom_idx];


                int g_atom_idx_k = mol_offsets[mol_idx]+k;

                const int an_j = atomic_nums[g_atom_idx_j];
                const int an_k = atomic_nums[g_atom_idx_k];

                float k_x = Xs[g_atom_idx_k];
                float k_y = Ys[g_atom_idx_k];
                float k_z = Zs[g_atom_idx_k];

                float d_ik_x = i_x - k_x;
                float d_ik_y = i_y - k_y;
                float d_ik_z = i_z - k_z;

                float r_ik = dist_diff(d_ik_x, d_ik_y, d_ik_z);

                if(r_ik < A_Rc) {

                    // TODO(YTZ): replace with arctan2 trick

                    float inner = (d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z) / (r_ij * r_ik);
                    inner = fmaxf(inner, -1.0);
                    inner = fminf(inner,  1.0);

                    // printf("INNER %f\n", inner);

                    float theta_ijk = acosf(inner);

                    // super useful debug
                    if(isnan(theta_ijk) || isinf(theta_ijk)) {
                        printf("WTF NAN/INF: %d, %d, %d, r_ij, r_ik, %f, %f, top %f, bottom %f, i_coords:(%f, %f, %f), j_coords(%f, %f, %f), k_coords(%f, %f, %f)\n",
                            g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, r_ij, r_ik, d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z, r_ij * r_ik, i_x, i_y, i_z, j_x, j_y, j_z, k_x, k_y, k_z);
                    }

                    // printf("gpu tijk %d %d %d %f\n", local_atom_idx, j, k, theta_ijk);
                    float A_f_C_ik = f_C(r_ik, A_Rc);
                    for(int t=0; t < NUM_A_THETAS; t++) {
                        for(int s=0; s < NUM_A_RS; s++) {
                            // (TODO: ytz) do 2*(1-A_Zeta) at the end
                            float summand = powf(2, 1-A_zeta) * powf(1+cosf(theta_ijk - A_thetas[t]), A_zeta) * expf(-A_eta*powf((r_ij + r_ik)/2 - A_Rs[s], 2)) * A_f_C_ij * A_f_C_ik;
                            // printf("summand: %f, \n", summand);
                            // printf("scatter_idxs[g_atom_idx_i]: %d, linearize: %d\n", scatter_idxs[g_atom_idx_i], linearize(an_j, an_k, t, s));
                            // printf("i,j,k,t,s %d %d %d %d %d %d\n", g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, at_idx, ar_idx, linearize(j_type, k_type, at_idx, ar_idx))
                            auto res = atomicAdd(angular_feature_buffer_i + linearize(an_j, an_k, t, s), summand);


                            // if(isnan(res) || isinf(res)) {
                            //     printf("WTF ANGULAR SUMMAND NAN/INF: %d, %d, %d, r_ij, r_ik, %f, %f, top %f, bottom %f, i_coords:(%f, %f, %f), j_coords(%f, %f, %f), k_coords(%f, %f, %f)\n",
                            //         g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, r_ij, r_ik, d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z, r_ij * r_ik, i_x, i_y, i_z, j_x, j_y, j_z, k_x, k_y, k_z);
                            // }
                        }
                    }     
                }
            }
        }
    }
}


template<typename T>
T *cudaMallocSimple(size_t n) {
    T *d_obj;

    std::cout << "mallocing:" << n*sizeof(T) << "bytes\n";
    assert(cudaMalloc(&d_obj, n*sizeof(T)) == 0);
    return d_obj;
}

template<typename T>
void cudaCopySimple(T *obj, size_t n, T *d_obj) {
    assert(cudaMemcpy(d_obj, obj, n*sizeof(T), cudaMemcpyHostToDevice) == 0);
}

typedef std::chrono::high_resolution_clock Clock;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}   

// int main(void) {

//     auto X_obj = cnpy::npy_load("Xs.npy");
//     auto Y_obj = cnpy::npy_load("Ys.npy");
//     auto Z_obj = cnpy::npy_load("Zs.npy");
//     auto A_obj = cnpy::npy_load("As.npy");

//     auto MOs = cnpy::npy_load("MOs.npy");
//     auto MACs = cnpy::npy_load("MACs.npy");

//     float *d_Xs = cudaMallocSimple<float>(X_obj.shape[0]);
//     float *d_Ys = cudaMallocSimple<float>(Y_obj.shape[0]);
//     float *d_Zs = cudaMallocSimple<float>(Z_obj.shape[0]);
//     int *d_As = cudaMallocSimple<int>(A_obj.shape[0]);
//     int *d_MOs = cudaMallocSimple<int>(MOs.shape[0]);
//     int *d_MACs = cudaMallocSimple<int>(MACs.shape[0]); // max

//     size_t n_total_atoms = X_obj.shape[0];
//     size_t n_mols  = MOs.shape[0];


//     int sort_num_items = n_total_atoms; // change to upperbound later, max number of atoms per block

//     int *d_vals_in = cudaMallocSimple<int>(sort_num_items);
//     int *sort_idxs = cudaMallocSimple<int>(sort_num_items);
//     int *inv_idxs  = cudaMallocSimple<int>(sort_num_items);

//     std::vector<int> idxs(sort_num_items);

//     for(size_t i=0; i < sort_num_items; i++) {
//         idxs[i] = i;
//     }

//     cudaCopySimple(&idxs[0], sort_num_items, d_vals_in);
//     int *d_keys_out = cudaMallocSimple<int>(sort_num_items);
//     void *d_temp_storage = NULL;
//     size_t temp_storage_bytes = 0;

//     // determine size requirements
//     gpuErrchk(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_As, d_keys_out, d_vals_in, sort_idxs, sort_num_items));
//     gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes)) ;

//     // SETUP DONE

//     float* d_X_feat;

//     printf("Total number of atoms: %d \n", n_total_atoms);

//     std::cout << "mallocing:" << n_total_atoms*TOTAL_FEATURE_SIZE*sizeof(float) << "bytes\n";

//     cudaMalloc(&d_X_feat, n_total_atoms*TOTAL_FEATURE_SIZE*sizeof(float)); 

//     auto start = Clock::now();




//     // int i=0;
//     for(size_t i=0; i < 100000; i++) {

//         int num_items = n_total_atoms; // upper bound this to a fixed num

//         std::cout << i << std::endl;    
//         cudaCopySimple(X_obj.data<float>(), X_obj.shape[0], d_Xs);
//         cudaCopySimple(Y_obj.data<float>(), Y_obj.shape[0], d_Ys);
//         cudaCopySimple(Z_obj.data<float>(), Z_obj.shape[0], d_Zs);
//         cudaCopySimple(A_obj.data<int>(), A_obj.shape[0], d_As);

//         cudaCopySimple(MOs.data<int>(), MOs.shape[0], d_MOs);
//         cudaCopySimple(MACs.data<int>(), MACs.shape[0], d_MACs); // max

//         assert(cudaMemset(d_X_feat, 0, n_total_atoms*TOTAL_FEATURE_SIZE*sizeof(float)) == 0);

//         // atom type counters
//         std::vector<int> counts(MAX_ATOM_TYPES, 0);

//         for(size_t j=0; j < n_total_atoms; j++) {
//             counts[A_obj.data<int>()[j]] += 1;
//         }


//         // 1. Sort by atom pairs.
//         // 2. 

//         // GPU
//         gpuErrchk(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_As, d_keys_out, d_vals_in, sort_idxs, sort_num_items));
//         inverse<<<n_mols, 32>>>(sort_idxs, inv_idxs, sort_num_items); // invert
//         gpuErrchk(cudaPeekAtLastError());



//         // follow up with a segment reduce

//         // std::vector<int> test(sort_num_items);
//         // cudaMemcpy(&test[0], inv_idxs, n_total_atoms*sizeof(int), cudaMemcpyDeviceToHost);

//         // for(auto v : test) {
//         //     std::cout << v << " ";
//         // }

//         // return;



//         // CPU
//         // std::vector<int> buffer(A_obj.data<int>(), A_obj.data<int>() + A_obj.shape[0]);
//         // std::vector<int> h_sort_idx = sort_indexes(buffer);
//         // for(size_t k=0; k < sort_num_items; k++) {
//         //     buffer[h_sort_idx[k]] = k;
//         // }
//         // cudaCopySimple(&buffer[0], sort_num_items, inv_idxs);


//         //START
//         featurize<<<n_mols, 32>>>(
//             d_Xs,
//             d_Ys,
//             d_Zs,
//             d_As,
//             d_MOs,
//             d_MACs,
//             n_mols,
//             inv_idxs,
//             d_X_feat);


//         gpuErrchk( cudaPeekAtLastError() );

//         auto end = Clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();


//         std::cout << "DURATION:" << duration << std::endl;

//         std::cout << "samples per minute:" << (float((i+1)*n_mols) / duration) * 60 * 1e9 << std::endl;

//     }

//     std::vector<float> X_feat(n_total_atoms*TOTAL_FEATURE_SIZE, 0);
//     cudaMemcpy(&X_feat[0], d_X_feat, n_total_atoms*TOTAL_FEATURE_SIZE*sizeof(int), cudaMemcpyDeviceToHost);





// }