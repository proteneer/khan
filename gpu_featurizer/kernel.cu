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

    return sqrtf(dx*dx+dy*dy+dz*dz);

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
inline __device__ int linearize(int i, int j, int k, int l, AniParams params) {
    if(j < i) {
        float tmp = i;
        i = j;
        j = tmp;
    }

    const auto N = params.max_types;
    const auto K = params.Num_A_thetas;
    const auto L = params.Num_A_Rs;

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
__global__ void featurize_gpu(
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
    AniParams params) {

    int mol_idx = blockIdx.x;
    int num_atoms = mol_atom_count[blockIdx.x];
    int block_size = blockDim.x;
    int num_warps = (num_atoms + block_size - 1)/block_size; // how many warps we need to process

    const size_t angular_feature_size = params.Num_A_Rs * params.Num_A_thetas * (params.max_types * (params.max_types+1) / 2);
    const size_t radial_feature_size = params.max_types * params.Num_R_Rs;
    const size_t total_feature_size = angular_feature_size + radial_feature_size;

    for(int warp_idx = 0; warp_idx < num_warps; warp_idx++) {

        int local_atom_idx = warp_idx*block_size + threadIdx.x; // local_local_atom_idx

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
      
        float *radial_feature_buffer_i = X_feat_out_i + scatter_idxs[g_atom_idx_i]*total_feature_size + 0;
        float *angular_feature_buffer_i = X_feat_out_i + scatter_idxs[g_atom_idx_i]*total_feature_size + radial_feature_size;

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

            float *radial_feature_buffer_j = X_feat_out_j + scatter_idxs[g_atom_idx_j]*total_feature_size + 0;
            // float *radial_feature_buffer_j = X_feat_j + g_atom_idx_j*TOTAL_FEATURE_SIZE + 0;
            // float *angular_feature_buffer_j = X_feat_out + g_atom_idx_j*TOTAL_FEATURE_SIZE + RADIAL_FEATURE_SIZE;


            // if(g_atom_idx_i == 0) {
                // printf("gpu j %d %f\n", j, r_ij);
                // printf("summand, offset, %f, %d\n", summand, scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + atomic_nums[g_atom_idx_j] * params.Num_R_Rs + r_idx);
                // printf("summand, offset, %f, %d\n", summand, scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + atomic_nums[g_atom_idx_j] * params.Num_R_Rs + r_idx);
            // }

            // radial features
            if(r_ij < params.R_Rc && local_atom_idx < j) {
                for(int r_idx = 0; r_idx < params.Num_R_Rs; r_idx++) {
                    float diff = r_ij - params.R_Rs[r_idx];
                    float summand = expf(-params.R_eta * diff * diff) * f_C(r_ij, params.R_Rc);

                    // // float inner = powf(r_ij - params.R_Rs[r_idx], 2);


                    // // if(isnan(inner) || isinf(inner)) {
                    // //     printf("SUMMAND INNER NAN\n");
                    // // }



                    // float diff = r_ij - params.R_Rs[r_idx];

                    // float inner = diff*diff;


                    // if(isnan(inner) || isinf(inner)) {
                    //     printf("SUMMAND INNER NAN %f %f %f\n", r_ij,  params.R_Rs[r_idx], inner);
                    // }

                    // float a = expf(-params.R_eta * inner);
                    // float b = f_C(r_ij, params.R_Rc);

                    // if(isnan(a) || isinf(a)) {
                    //     printf("SUMMAND A NAN\n");
                    // }

                    // if(isnan(b) || isinf(b)) {
                    //     printf("SUMMAND B NAN\n");
                    // }


                    // float summand = a*b;


                    // exploit symmetry of the atomic adds
                    auto res1 = atomicAdd(radial_feature_buffer_i + atomic_nums[g_atom_idx_j] * params.Num_R_Rs + r_idx, summand);
                    auto res2 = atomicAdd(radial_feature_buffer_j + atomic_nums[g_atom_idx_i] * params.Num_R_Rs + r_idx, summand);

                    // if(isnan(res1) || isinf(res1)) {
                       // printf("WTF RADIAL RES1 NAN/INF, offset, %f, %f\n", res1, summand);
                        // : %d, %d, %d, r_ij, r_ik, %f, %f, top %f, bottom %f, i_coords:(%f, %f, %f), j_coords(%f, %f, %f), k_coords(%f, %f, %f)\n",
                            // g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, r_ij, r_ik, d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z, r_ij * r_ik, i_x, i_y, i_z, j_x, j_y, j_z, k_x, k_y, k_z);
                    // }

                    // if(isnan(res2) || isinf(res2)) {
                       // printf("WTF RADIAL RES2 NAN/INF, offset, %f, %f\n", res2, summand);
                        // : %d, %d, %d, r_ij, r_ik, %f, %f, top %f, bottom %f, i_coords:(%f, %f, %f), j_coords(%f, %f, %f), k_coords(%f, %f, %f)\n",
                            // g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, r_ij, r_ik, d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z, r_ij * r_ik, i_x, i_y, i_z, j_x, j_y, j_z, k_x, k_y, k_z);
                    // }

                }
            }

            float A_f_C_ij = f_C(r_ij, params.A_Rc);


            if(r_ij < params.A_Rc) {

                for(size_t k=j+1; k < num_atoms; k++) {

                    if(local_atom_idx == j || local_atom_idx == k || j == k) {
                        continue;
                    }

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

                    if(r_ik < params.A_Rc) {

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
                        float A_f_C_ik = f_C(r_ik, params.A_Rc);
                        for(int t=0; t < params.Num_A_thetas; t++) {
                            for(int s=0; s < params.Num_A_Rs; s++) {
                                // (TODO: ytz) do 2*(1-A_Zeta) at the end
                                float inner = (r_ij + r_ik)/2 - params.A_Rs[s]; // powf(x,y) is numerically unstable for negative xs so we simply double them.
                                float summand = powf(2, 1-params.A_zeta) * powf(1+cosf(theta_ijk - params.A_thetas[t]), params.A_zeta) * expf(-params.A_eta*inner*inner) * A_f_C_ij * A_f_C_ik;
                                // printf("summand: %f, \n", summand);
                                // printf("scatter_idxs[g_atom_idx_i]: %d, linearize: %d\n", scatter_idxs[g_atom_idx_i], linearize(an_j, an_k, t, s));
                                // printf("i,j,k,t,s %d %d %d %d %d %d\n", g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, at_idx, ar_idx, linearize(j_type, k_type, at_idx, ar_idx))
                                auto res = atomicAdd(angular_feature_buffer_i + linearize(an_j, an_k, t, s, params), summand);

                                // if(isnan(res) || isinf(res)) {
                                //     printf("WTF ANGULAR SUMMAND NAN/INF: %d, %d, %d, r_ij, r_ik, %f, %f, top %f, bottom %f, i_coords:(%f, %f, %f), j_coords(%f, %f, %f), k_coords(%f, %f, %f)\n",
                                //         g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, r_ij, r_ik, d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z, r_ij * r_ik, i_x, i_y, i_z, j_x, j_y, j_z, k_x, k_y, k_z);
                                // }
                            }
                        }
                    }
                }
            } // end radial
        } // end current warp
    } // end all warps

}

// template<typename X_type, typename Y_type>
__device__ static inline float square(float x) {
    return x*x;
}

#include "assert.h"

// Remind yutong to document what these pointers are.
__global__ void featurize_grad_gpu(
    const float *Xs,
    const float *Ys,
    const float *Zs,

    const int *atomic_nums,
    const int *mol_offsets,
    const int *mol_atom_count,
    const int num_mols, // actually equal to blockDim.x

    const int *scatter_idxs, // LOCAL WITHIN THE ATOM TYPE

    const float *H_grads,
    const float *C_grads,
    const float *N_grads,
    const float *O_grads,

    float *X_grads,
    float *Y_grads,
    float *Z_grads,

    AniParams params) {

    int mol_idx = blockIdx.x;
    int num_atoms = mol_atom_count[blockIdx.x];
    int block_size = blockDim.x;
    int num_warps = (num_atoms + block_size - 1)/block_size; // how many warps we need to process

    const size_t angular_feature_size = params.Num_A_Rs * params.Num_A_thetas * (params.max_types * (params.max_types+1) / 2);
    const size_t radial_feature_size = params.max_types * params.Num_R_Rs;
    const size_t total_feature_size = angular_feature_size + radial_feature_size;

    for(int warp_idx = 0; warp_idx < num_warps; warp_idx++) {

        int local_atom_idx = warp_idx*block_size + threadIdx.x; // local_local_atom_idx

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

        const float *output_buffer_i;
        if(g_atomic_num_i == 0) {
            output_buffer_i = H_grads;
        } else if(g_atomic_num_i == 1) {
            output_buffer_i = C_grads;
        } else if(g_atomic_num_i == 2) {
            output_buffer_i = N_grads;
        } else {
            output_buffer_i = O_grads;
        }
      
        const float *radial_feature_buffer_i = output_buffer_i + scatter_idxs[g_atom_idx_i]*total_feature_size + 0;
        const float *angular_feature_buffer_i = output_buffer_i + scatter_idxs[g_atom_idx_i]*total_feature_size + radial_feature_size;

        for(int j=0; j < num_atoms; j++) {

            int g_atom_idx_j = mol_offsets[mol_idx]+j;
            // int g_atomic_num_j = atomic_nums[g_atom_idx_j];

            float j_x = Xs[g_atom_idx_j];
            float j_y = Ys[g_atom_idx_j];
            float j_z = Zs[g_atom_idx_j];

            float d_ij_x = i_x - j_x;
            float d_ij_y = i_y - j_y;
            float d_ij_z = i_z - j_z;

            // printf("(%f, %f, %f)\n", d_ij_x, d_ij_y, d_ij_z);

            float r_ij = dist_diff(d_ij_x, d_ij_y, d_ij_z);

            // radial features
            if(r_ij < params.R_Rc && local_atom_idx != j) {
                for(int r_idx = 0; r_idx < params.Num_R_Rs; r_idx++) {

                    float d_y_i = radial_feature_buffer_i[atomic_nums[g_atom_idx_j] * params.Num_R_Rs + r_idx]; // gradient broadcast

                    // todo: convert cos to cosf

                    float accum_i_x = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(i_x - j_x)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))/r_ij - 0.5*M_PI*(i_x - j_x)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);
                    float accum_i_y = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(i_y - j_y)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))/r_ij - 0.5*M_PI*(i_y - j_y)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);
                    float accum_i_z = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(i_z - j_z)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))/r_ij - 0.5*M_PI*(i_z - j_z)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);

                    // accumulate locally
                    atomicAdd(X_grads+g_atom_idx_i, accum_i_x * d_y_i);
                    atomicAdd(Y_grads+g_atom_idx_i, accum_i_y * d_y_i);
                    atomicAdd(Z_grads+g_atom_idx_i, accum_i_z * d_y_i);

                    float accum_j_x = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(-i_x + j_x)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))/r_ij - 0.5*M_PI*(-i_x + j_x)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);
                    float accum_j_y = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(-i_y + j_y)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))/r_ij - 0.5*M_PI*(-i_y + j_y)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);
                    float accum_j_z = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(-i_z + j_z)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))/r_ij - 0.5*M_PI*(-i_z + j_z)*expf(-params.R_eta*square(-params.R_Rs[r_idx] + r_ij))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);

                    atomicAdd(X_grads+g_atom_idx_j, accum_j_x * d_y_i);
                    atomicAdd(Y_grads+g_atom_idx_j, accum_j_y * d_y_i);
                    atomicAdd(Z_grads+g_atom_idx_j, accum_j_z * d_y_i);

                }
            }

            float A_f_C_ij = f_C(r_ij, params.A_Rc);


            if(r_ij < params.A_Rc) {
                for(size_t k=j+1; k < num_atoms; k++) {
                    if(local_atom_idx == j || local_atom_idx == k || j == k) {
                        continue;
                    }

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


                    if(r_ik < params.A_Rc) {
                        for(int t=0; t < params.Num_A_thetas; t++) {
                            for(int s=0; s < params.Num_A_Rs; s++) {

                                float d_y_i = angular_feature_buffer_i[linearize(an_j, an_k, t, s, params)];   

                                if(isnan(d_y_i) || isinf(d_y_i)) {

                                    printf("garbage_d_y_i detected! \n");
                                    assert(0);
                                }

                                float l2ij = square(i_x - j_x) + square(i_y - j_y) + square(i_z - j_z);
                                float l2ik = square(i_x - k_x) + square(i_y - k_y) + square(i_z - k_z);
                                float dij = sqrtf(l2ij);
                                float dik = sqrtf(l2ik);
                                // rij . rik
                                float ijk_swizzle = (i_x - j_x)*(i_x - k_x) + (i_y - j_y)*(i_y - k_y) + (i_z - j_z)*(i_z - k_z);

                                float cos_theta = (ijk_swizzle)/(dij*dik);

                                float eps = 1e-6;

                                //skipping if 0 or pi
                                if((abs(cos_theta - 1) < eps) || (abs(cos_theta + 1) < eps)) {
                                    continue;
                                }
                                // angle between ijk is zero or 180
                                if(acosf(cos_theta) == 0) {
                                    continue;
                                }

                                // if(cos_theta > 1.0 || cos_theta < -1.0) {
                                //     printf("shitty cos_theta %f\n", cos_theta);
                                //     assert(0);
                                // }
                                float dtheta = cosf(params.A_thetas[t] - acosf(cos_theta)) + 1;


                                // float dotnorm = powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2)));


                                // auto aaa = -powf(2.0, -params.A_zeta + 1)*params.A_eta*((i_x - j_x)/dij + (i_x - k_x)/dik)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2));
                                // float bbb = powf(2.0, -params.A_zeta + 1)*params.A_zeta*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*((-i_x + j_x)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik) + (-i_x + k_x)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)) + (2*i_x - j_x - k_x)/(dij*dik));
                                // float ccc = expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta));
                                // float ccc_denom = (sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1));
                                // float ddd = 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_x - j_x)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc);
                                // float ddd_denom = (params.A_Rc*dij);


                                // float accum_i_x = aaa - bbb*ccc/ccc_denom - ddd/ddd_denom - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_x - k_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);
                                float accum_i_x = -powf(2.0, -params.A_zeta + 1)*params.A_eta*((i_x - j_x)/dij + (i_x - k_x)/dik)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - powf(2.0, -params.A_zeta + 1)*params.A_zeta*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*((-i_x + j_x)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik) + (-i_x + k_x)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)) + (2*i_x - j_x - k_x)/(dij*dik))*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_x - j_x)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_x - k_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);
                                float accum_i_y = -powf(2.0, -params.A_zeta + 1)*params.A_eta*((i_y - j_y)/dij + (i_y - k_y)/dik)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - powf(2.0, -params.A_zeta + 1)*params.A_zeta*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*((-i_y + j_y)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik) + (-i_y + k_y)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)) + (2*i_y - j_y - k_y)/(dij*dik))*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_y - j_y)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_y - k_y)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);
                                float accum_i_z = -powf(2.0, -params.A_zeta + 1)*params.A_eta*((i_z - j_z)/dij + (i_z - k_z)/dik)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - powf(2.0, -params.A_zeta + 1)*params.A_zeta*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*((-i_z + j_z)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik) + (-i_z + k_z)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)) + (2*i_z - j_z - k_z)/(dij*dik))*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_z - j_z)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_z - k_z)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);

                                float accum_j_x = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_x + j_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_x + k_x)/(dij*dik) + (i_x - j_x)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_x + j_x)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij);
                                float accum_j_y = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_y + j_y)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_y + k_y)/(dij*dik) + (i_y - j_y)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_y + j_y)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij);
                                float accum_j_z = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_z + j_z)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_z + k_z)/(dij*dik) + (i_z - j_z)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_z + j_z)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij); 

                                float accum_k_x = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_x + k_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_x + j_x)/(dij*dik) + (i_x - k_x)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_x + k_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);
                                float accum_k_y = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_y + k_y)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_y + j_y)/(dij*dik) + (i_y - k_y)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_y + k_y)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);
                                float accum_k_z = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_z + k_z)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_z + j_z)/(dij*dik) + (i_z - k_z)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf(cos_theta))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_z + k_z)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);

                                if(isnan(accum_i_x) || isinf(accum_i_x)) {
                                    // auto aaa = -powf(2.0, -params.A_zeta + 1)*params.A_eta*((i_x - j_x)/dij);
                                    // auto bbb = (0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5);
                                    // auto ccc = powf(cosf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik))) + 1, params.A_zeta);
                                    // auto ddd = (-params.A_Rs[s] + (0.5)*dij + (0.5)*dik);
                                    // auto eee = expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2));
                                    // auto fff = powf(2.0, -params.A_zeta + 1)*params.A_zeta*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5);

                                    // auto ggg = powf(cosf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik))) + 1, params.A_zeta);

                                    // auto hhh = ((-i_x + j_x)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik) + (-i_x + k_x)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)) + (2*i_x - j_x - k_x)/(dij*dik));

                                    // auto hihi = expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)));

                                    // auto iii = sqrt(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)*(cosf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik))) + 1);



                                    // auto jjj = 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_x - j_x)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(cosf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik))) + 1, params.A_zeta);
                                    // auto kkk = expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij);
                                    // auto lll = 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_x - k_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(cosf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik))) + 1, params.A_zeta);
                                    // auto mmm = expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);


                                    // printf("garbage accum_i_x detected! %f %f %f %f %f %f %f %f\n", dtheta, acosf(cos_theta), aaa, bbb, ccc, ccc_denom, ddd, ddd_denom);
                                    // assert(0);
                                }

                                // if(isnan(accum_j_x) || isinf(accum_j_x)) {
                                //     printf("garbage accum_j_x detected! \n");
                                //     assert(0);
                                // }
                                // if(isnan(accum_k_x) || isinf(accum_k_x)) {
                                //     printf("garbage accum_k_x detected! \n");
                                //     assert(0);
                                // }

                                // if(isnan(accum_i_y) || isinf(accum_i_y)) {
                                //     printf("garbage accum_i_y detected! \n");
                                //     assert(0);
                                // }
                                // if(isnan(accum_j_y) || isinf(accum_j_y)) {
                                //     printf("garbage accum_j_y detected! \n");
                                //     assert(0);
                                // }
                                // if(isnan(accum_k_y) || isinf(accum_k_y)) {
                                //     printf("garbage accum_k_y detected! \n");
                                //     assert(0);
                                // }

                                // if(isnan(accum_i_z) || isinf(accum_i_z)) {
                                //     printf("garbage accum_i_z detected! \n");
                                //     assert(0);
                                // }
                                // if(isnan(accum_j_z) || isinf(accum_j_z)) {
                                //     printf("garbage accum_j_z detected! \n");
                                //     assert(0);
                                // }
                                // if(isnan(accum_k_z) || isinf(accum_k_z)) {
                                //     printf("garbage accum_k_z detected! \n");
                                //     assert(0);
                                // }

                                // these are accumulated
                                atomicAdd(X_grads+g_atom_idx_i, accum_i_x * d_y_i);
                                atomicAdd(Y_grads+g_atom_idx_i, accum_i_y * d_y_i);
                                atomicAdd(Z_grads+g_atom_idx_i, accum_i_z * d_y_i);

                                atomicAdd(X_grads+g_atom_idx_j, accum_j_x * d_y_i);
                                atomicAdd(Y_grads+g_atom_idx_j, accum_j_y * d_y_i);
                                atomicAdd(Z_grads+g_atom_idx_j, accum_j_z * d_y_i);

                                atomicAdd(X_grads+g_atom_idx_k, accum_k_x * d_y_i);
                                atomicAdd(Y_grads+g_atom_idx_k, accum_k_y * d_y_i);
                                atomicAdd(Z_grads+g_atom_idx_k, accum_k_z * d_y_i);
                            }
                        }
                    }
                }
            } // end radial
        } // end current warp
    } // end all warps

}

// Remind yutong to document what these pointers are.
__global__ void featurize_grad_inverse_gpu(
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

    float *H_grads,
    float *C_grads,
    float *N_grads,
    float *O_grads,

    AniParams params) {

    int mol_idx = blockIdx.x;
    int num_atoms = mol_atom_count[blockIdx.x];
    int block_size = blockDim.x;
    int num_warps = (num_atoms + block_size - 1)/block_size; // how many warps we need to process

    const size_t angular_feature_size = params.Num_A_Rs * params.Num_A_thetas * (params.max_types * (params.max_types+1) / 2);
    const size_t radial_feature_size = params.max_types * params.Num_R_Rs;
    const size_t total_feature_size = angular_feature_size + radial_feature_size;

    for(int warp_idx = 0; warp_idx < num_warps; warp_idx++) {

        int local_atom_idx = warp_idx*block_size + threadIdx.x; // local_local_atom_idx

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

        float *output_buffer_i;
        if(g_atomic_num_i == 0) {
            output_buffer_i = H_grads;
        } else if(g_atomic_num_i == 1) {
            output_buffer_i = C_grads;
        } else if(g_atomic_num_i == 2) {
            output_buffer_i = N_grads;
        } else {
            output_buffer_i = O_grads;
        }
      
        float *radial_feature_buffer_i = output_buffer_i + scatter_idxs[g_atom_idx_i]*total_feature_size + 0;
        float *angular_feature_buffer_i = output_buffer_i + scatter_idxs[g_atom_idx_i]*total_feature_size + radial_feature_size;

        for(int j=0; j < num_atoms; j++) {

            int g_atom_idx_j = mol_offsets[mol_idx]+j;
            // int g_atomic_num_j = atomic_nums[g_atom_idx_j];

            float j_x = Xs[g_atom_idx_j];
            float j_y = Ys[g_atom_idx_j];
            float j_z = Zs[g_atom_idx_j];

            float d_ij_x = i_x - j_x;
            float d_ij_y = i_y - j_y;
            float d_ij_z = i_z - j_z;

            // printf("(%f, %f, %f)\n", d_ij_x, d_ij_y, d_ij_z);

            float r_ij = dist_diff(d_ij_x, d_ij_y, d_ij_z);

            // radial features
            if(r_ij < params.R_Rc && local_atom_idx != j) {
                for(int r_idx = 0; r_idx < params.Num_R_Rs; r_idx++) {
                        float accum_i_x = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(i_x - j_x)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(i_x - j_x)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);
                        float accum_i_y = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(i_y - j_y)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(i_y - j_y)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);
                        float accum_i_z = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(i_z - j_z)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(i_z - j_z)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);

                        float accumulant = accum_i_x * X_grads[g_atomic_num_i] + accum_i_y * Y_grads[g_atomic_num_i] + accum_i_z * Z_grads[g_atomic_num_i];
                        radial_feature_buffer_i[atomic_nums[g_atom_idx_j] * params.Num_R_Rs + r_idx] += accumulant;

                        float accum_j_x = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(-i_x + j_x)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(-i_x + j_x)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);
                        float accum_j_y = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(-i_y + j_y)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(-i_y + j_y)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);
                        float accum_j_z = -2*params.R_eta*(-params.R_Rs[r_idx] + r_ij)*(-i_z + j_z)*(0.5*cosf(M_PI*r_ij/params.R_Rc) + 0.5)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(-i_z + j_z)*expf(-params.R_eta*powf(-params.R_Rs[r_idx] + r_ij, 2))*sinf(M_PI*r_ij/params.R_Rc)/(params.R_Rc*r_ij);

                        accumulant = accum_j_x * X_grads[g_atomic_num_i] + accum_j_y * Y_grads[g_atomic_num_i] + accum_j_z * Z_grads[g_atomic_num_i];
                        atomicAdd(radial_feature_buffer_i + atomic_nums[g_atom_idx_j] * params.Num_R_Rs + r_idx, accumulant);
                }
            }

            float A_f_C_ij = f_C(r_ij, params.A_Rc);


            if(r_ij < params.A_Rc) {
                for(size_t k=j+1; k < num_atoms; k++) {
                    if(local_atom_idx == j || local_atom_idx == k || j == k) {
                        continue;
                    }

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

                    if(r_ik < params.A_Rc) {
                        for(int t=0; t < params.Num_A_thetas; t++) {
                            for(int s=0; s < params.Num_A_Rs; s++) {
                                    float dij = sqrtf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2));
                                    float dik = sqrtf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2));
                                    float ijk_swizzle = (i_x - j_x)*(i_x - k_x) + (i_y - j_y)*(i_y - k_y) + (i_z - j_z)*(i_z - k_z);

                                    float dtheta = cosf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik))) + 1;

                                    float cos_theta = (ijk_swizzle)/(dij*dik);

                                    float eps = 1e-6;

                                    //skipping if 0 or pi
                                    if((abs(cos_theta - 1) < eps) || (abs(cos_theta + 1) < eps)) {
                                        continue;
                                    }

                                    float accum_i_x = -powf(2.0, -params.A_zeta + 1)*params.A_eta*((i_x - j_x)/dij + (i_x - k_x)/dik)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - powf(2.0, -params.A_zeta + 1)*params.A_zeta*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*((-i_x + j_x)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik) + (-i_x + k_x)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)) + (2*i_x - j_x - k_x)/(dij*dik))*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_x - j_x)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_x - k_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);
                                    float accum_i_y = -powf(2.0, -params.A_zeta + 1)*params.A_eta*((i_y - j_y)/dij + (i_y - k_y)/dik)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - powf(2.0, -params.A_zeta + 1)*params.A_zeta*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*((-i_y + j_y)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik) + (-i_y + k_y)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)) + (2*i_y - j_y - k_y)/(dij*dik))*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_y - j_y)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_y - k_y)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);
                                    float accum_i_z = -powf(2.0, -params.A_zeta + 1)*params.A_eta*((i_z - j_z)/dij + (i_z - k_z)/dik)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - powf(2.0, -params.A_zeta + 1)*params.A_zeta*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*((-i_z + j_z)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik) + (-i_z + k_z)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)) + (2*i_z - j_z - k_z)/(dij*dik))*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_z - j_z)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(i_z - k_z)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);

                                    float accum_j_x = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_x + j_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_x + k_x)/(dij*dik) + (i_x - j_x)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_x + j_x)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij);
                                    float accum_j_y = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_y + j_y)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_y + k_y)/(dij*dik) + (i_y - j_y)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_y + j_y)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij);
                                    float accum_j_z = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_z + j_z)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_z + k_z)/(dij*dik) + (i_z - j_z)*(ijk_swizzle)/(powf(powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2), 1.5)*dik))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_z + j_z)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dij/params.A_Rc)/(params.A_Rc*dij); 

                                    float accum_k_x = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_x + k_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_x + j_x)/(dij*dik) + (i_x - k_x)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_x + k_x)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);
                                    float accum_k_y = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_y + k_y)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_y + j_y)/(dij*dik) + (i_y - k_y)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_y + k_y)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);
                                    float accum_k_z = -powf(2.0, -params.A_zeta + 1)*params.A_eta*(-i_z + k_z)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - powf(2.0, -params.A_zeta + 1)*params.A_zeta*((-i_z + j_z)/(dij*dik) + (i_z - k_z)*(ijk_swizzle)/(dij*powf(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2), 1.5)))*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*(0.5*cosf(M_PI*dik/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta-1)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(params.A_thetas[t] - acosf((ijk_swizzle)/(dij*dik)))/(sqrtf(-powf(ijk_swizzle, 2)/((powf(i_x - j_x, 2) + powf(i_y - j_y, 2) + powf(i_z - j_z, 2))*(powf(i_x - k_x, 2) + powf(i_y - k_y, 2) + powf(i_z - k_z, 2))) + 1)) - 0.5*powf(2.0, -params.A_zeta + 1)*M_PI*(-i_z + k_z)*(0.5*cosf(M_PI*dij/params.A_Rc) + 0.5)*powf(dtheta, params.A_zeta)*expf(-params.A_eta*powf(-params.A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sinf(M_PI*dik/params.A_Rc)/(params.A_Rc*dik);

                                    float accumulant = accum_i_x * X_grads[g_atomic_num_i] + accum_i_y * Y_grads[g_atomic_num_i] + accum_i_z * Z_grads[g_atomic_num_i];
                                    atomicAdd(angular_feature_buffer_i + linearize(an_j, an_k, t, s, params), accumulant);

                                    accumulant = accum_j_x * X_grads[g_atomic_num_i] + accum_j_y * Y_grads[g_atomic_num_i] + accum_j_z * Z_grads[g_atomic_num_i];
                                    atomicAdd(angular_feature_buffer_i + linearize(an_j, an_k, t, s, params), accumulant);

                                    accumulant = accum_k_x * X_grads[g_atomic_num_i] + accum_k_y * Y_grads[g_atomic_num_i] + accum_k_z * Z_grads[g_atomic_num_i];
                                    atomicAdd(angular_feature_buffer_i + linearize(an_j, an_k, t, s, params), accumulant);
                            }
                        }
                    }
                }
            } // end radial
        } // end current warp
    } // end all warps

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
