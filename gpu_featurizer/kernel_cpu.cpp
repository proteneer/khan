#include <math.h>
#include "parameters.h"

static inline float dist_diff(float dx, float dy, float dz) {

    return sqrt(dx*dx+dy*dy+dz*dz);

}

static inline float f_C(float r_ij, float r_c) {
    if (r_ij <= r_c) {
        return 0.5 * cosf((M_PI * r_ij) / r_c) + 0.5;
    } else {
        return 0;
    }
}


static inline int linearize(int i, int j, int k, int l) {
    if(j < i) {
        float tmp = i;
        i = j;
        j = tmp;
    }

    const int N = MAX_ATOM_TYPES;
    const int K = NUM_A_THETAS;
    const int L = NUM_A_RS;

    int basis = (N*(N-1)/2 - (N-i) * (N-i-1)/2 +j);
    
    return basis*K*L + k*L + l;
}


void featurize_cpu(
    const float *input_Xs,
    const float *input_Ys,
    const float *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols,
    const int *scatter_idxs,
    float *X_feat_out_H,
    float *X_feat_out_C,
    float *X_feat_out_N,
    float *X_feat_out_O) {
 
    // std::cout << "start featurize_cpu" << std::endl;

    for(int mol_idx=0; mol_idx < n_mols; mol_idx++) {

        int num_atoms = input_MACs[mol_idx];

        for(int i_idx = 0; i_idx < num_atoms; i_idx++) {

            int g_atom_idx_i = mol_offsets[mol_idx] + i_idx;
            int g_atomic_num_i = input_As[g_atom_idx_i];

            float i_x = input_Xs[g_atom_idx_i];
            float i_y = input_Ys[g_atom_idx_i];
            float i_z = input_Zs[g_atom_idx_i];

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

            for(int j_idx = 0; j_idx < num_atoms; j_idx++) {
                int g_atom_idx_j = mol_offsets[mol_idx]+j_idx;
                int g_atomic_num_j = input_As[g_atom_idx_j];

                float j_x = input_Xs[g_atom_idx_j];
                float j_y = input_Ys[g_atom_idx_j];
                float j_z = input_Zs[g_atom_idx_j];

                float d_ij_x = i_x - j_x;
                float d_ij_y = i_y - j_y;
                float d_ij_z = i_z - j_z;

                float r_ij = dist_diff(d_ij_x, d_ij_y, d_ij_z);

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

                if(r_ij < R_Rc && i_idx < j_idx) {
                    for(int r_idx = 0; r_idx < NUM_R_Rs; r_idx++) {
                        float summand = expf(-R_eta * powf(r_ij - R_Rs[r_idx], 2.0)) * f_C(r_ij, R_Rc);
                        radial_feature_buffer_i[input_As[g_atom_idx_j] * NUM_R_Rs + r_idx] += summand;
                        radial_feature_buffer_j[input_As[g_atom_idx_i] * NUM_R_Rs + r_idx] += summand;
                    }
                }

                float A_f_C_ij = f_C(r_ij, A_Rc);

                if(r_ij < A_Rc) {
                    for(int k_idx = j_idx+1; k_idx < num_atoms; k_idx++) {
                        if(i_idx == j_idx || i_idx == k_idx || j_idx == k_idx) {
                            continue;
                        }
                        int g_atom_idx_k = mol_offsets[mol_idx]+k_idx;

                        const int an_j = input_As[g_atom_idx_j];
                        const int an_k = input_As[g_atom_idx_k];

                        float k_x = input_Xs[g_atom_idx_k];
                        float k_y = input_Ys[g_atom_idx_k];
                        float k_z = input_Zs[g_atom_idx_k];

                        float d_ik_x = i_x - k_x;
                        float d_ik_y = i_y - k_y;
                        float d_ik_z = i_z - k_z;

                        float r_ik = dist_diff(d_ik_x, d_ik_y, d_ik_z);

                        if(r_ik < A_Rc) {

                            // TODO(YTZ): replace with arctan2 trick
                            float inner = (d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z) / (r_ij * r_ik);
                            inner = fmaxf(inner, -1.0);
                            inner = fminf(inner,  1.0);
                            float theta_ijk = acosf(inner);
                            // super useful debug
                            // if(isnan(theta_ijk) || isinf(theta_ijk)) {
                                // printf("WTF NAN/INF: %d, %d, %d, r_ij, r_ik, %f, %f, top %f, bottom %f, i_coords:(%f, %f, %f), j_coords(%f, %f, %f), k_coords(%f, %f, %f)\n",
                                    // g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, r_ij, r_ik, d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z, r_ij * r_ik, i_x, i_y, i_z, j_x, j_y, j_z, k_x, k_y, k_z);
                            // }

                            float A_f_C_ik = f_C(r_ik, A_Rc);
                            for(int t=0; t < NUM_A_THETAS; t++) {
                                for(int s=0; s < NUM_A_RS; s++) {
                                    float summand = powf(2, 1-A_zeta) * powf(1+cosf(theta_ijk - A_thetas[t]), A_zeta) * expf(-A_eta*powf((r_ij + r_ik)/2 - A_Rs[s], 2)) * A_f_C_ij * A_f_C_ik;
                                    angular_feature_buffer_i[linearize(an_j, an_k, t, s)] += summand;
                                }
                            }     
                        }
                    }
                }
            }
        }
    }
}