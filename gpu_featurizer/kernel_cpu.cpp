

#include <iostream>

#include <math.h>
#include "parameters.h"
// #include "numeric_helpers.h"

static inline int linearize(int i, int j, int k, int l) {
    if(j < i) {
        int tmp = i;
        i = j;
        j = tmp;
    }

    const int N = MAX_ATOM_TYPES;
    const int K = NUM_A_THETAS;
    const int L = NUM_A_RS;

    int basis = (N*(N-1)/2 - (N-i) * (N-i-1)/2 +j);
    
    return basis*K*L + k*L + l;
}

static inline double dist_diff(double dx, double dy, double dz) {
    return sqrt(dx*dx+dy*dy+dz*dz);
}

static inline double f_C(double r_ij, double r_c) {
    if (r_ij <= r_c) {
        return 0.5 * cos((M_PI * r_ij) / r_c) + 0.5;
    } else {
        return 0;
    }
}

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
    NumericType *O_grads) {

    // std::cout << "decltype(i) is " << type_name<decltype(input_Xs)>() << '\n';
    // std::cout << "decltype(i) is " << type_name<decltype(X_grads)>() << '\n';
    // std::cout << "decltype(i) is " << type_name<decltype(H_grads)>() << '\n';

    // this is a loop over all unique pairs and triples
    for(int mol_idx=0; mol_idx < n_mols; mol_idx++) {

        int num_atoms = input_MACs[mol_idx];

        for(int i_idx = 0; i_idx < num_atoms; i_idx++) {

            int g_atom_idx_i = mol_offsets[mol_idx] + i_idx;
            int g_atomic_num_i = input_As[g_atom_idx_i];

            NumericType i_x = input_Xs[g_atom_idx_i];
            NumericType i_y = input_Ys[g_atom_idx_i];
            NumericType i_z = input_Zs[g_atom_idx_i];

            NumericType *X_grads_i, *Y_grads_i, *Z_grads_i;

            NumericType *input_buffer_i;
            if(g_atomic_num_i == 0) {
                input_buffer_i = H_grads;
            } else if(g_atomic_num_i == 1) {
                input_buffer_i = C_grads;
            } else if(g_atomic_num_i == 2) {
                input_buffer_i = N_grads;
            } else {
                input_buffer_i = O_grads;
            }

            NumericType *radial_feature_buffer_i = input_buffer_i + scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + 0;
            NumericType *angular_feature_buffer_i = input_buffer_i + scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + RADIAL_FEATURE_SIZE;

            for(int j_idx = 0; j_idx < num_atoms; j_idx++) {
                int g_atom_idx_j = mol_offsets[mol_idx]+j_idx;
                int g_atomic_num_j = input_As[g_atom_idx_j];

                NumericType j_x = input_Xs[g_atom_idx_j];
                NumericType j_y = input_Ys[g_atom_idx_j];
                NumericType j_z = input_Zs[g_atom_idx_j];

                NumericType d_ij_x = i_x - j_x;
                NumericType d_ij_y = i_y - j_y;
                NumericType d_ij_z = i_z - j_z;

                NumericType r_ij = dist_diff(d_ij_x, d_ij_y, d_ij_z);

                // note: the potential is computed as i_idx < j_idx but it double counts.
                if(r_ij < R_Rc && i_idx != j_idx) {
                    for(int r_idx = 0; r_idx < NUM_R_Rs; r_idx++) {
                        NumericType accum_i_x = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(i_x - j_x)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(i_x - j_x)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);
                        NumericType accum_i_y = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(i_y - j_y)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(i_y - j_y)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);
                        NumericType accum_i_z = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(i_z - j_z)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(i_z - j_z)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);

                        NumericType accumulant = accum_i_x * X_grads[g_atomic_num_i] + accum_i_y * Y_grads[g_atomic_num_i] + accum_i_z * Z_grads[g_atomic_num_i];
                        radial_feature_buffer_i[input_As[g_atom_idx_j] * NUM_R_Rs + r_idx] += accumulant;

                        NumericType accum_j_x = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(-i_x + j_x)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(-i_x + j_x)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);
                        NumericType accum_j_y = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(-i_y + j_y)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(-i_y + j_y)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);
                        NumericType accum_j_z = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(-i_z + j_z)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(-i_z + j_z)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);

                        accumulant = accum_j_x * X_grads[g_atomic_num_i] + accum_j_y * Y_grads[g_atomic_num_i] + accum_j_z * Z_grads[g_atomic_num_i];
                        radial_feature_buffer_i[input_As[g_atom_idx_j] * NUM_R_Rs + r_idx] += accumulant;
                    }
                }

                if(r_ij < A_Rc) {
                    for(int k_idx = j_idx+1; k_idx < num_atoms; k_idx++) {
                        if(i_idx == j_idx || i_idx == k_idx || j_idx == k_idx) {
                            continue;
                        }
                        int g_atom_idx_k = mol_offsets[mol_idx]+k_idx;
                        int g_atomic_num_k = input_As[g_atom_idx_k];

                        const int an_j = input_As[g_atom_idx_j];
                        const int an_k = input_As[g_atom_idx_k];

                        NumericType k_x = input_Xs[g_atom_idx_k];
                        NumericType k_y = input_Ys[g_atom_idx_k];
                        NumericType k_z = input_Zs[g_atom_idx_k];

                        NumericType d_ik_x = i_x - k_x;
                        NumericType d_ik_y = i_y - k_y;
                        NumericType d_ik_z = i_z - k_z;

                        NumericType r_ik = dist_diff(d_ik_x, d_ik_y, d_ik_z);

                        if(r_ik < A_Rc) {
                            for(int t=0; t < NUM_A_THETAS; t++) {
                                for(int s=0; s < NUM_A_RS; s++) {
                                    NumericType dij = sqrt(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2));
                                    NumericType dik = sqrt(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2));
                                    NumericType ijk_swizzle = (i_x - j_x)*(i_x - k_x) + (i_y - j_y)*(i_y - k_y) + (i_z - j_z)*(i_z - k_z);

                                    NumericType accum_i_x = -pow(2.0, -A_zeta + 1)*A_eta*((i_x - j_x)/dij + (i_x - k_x)/dik)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - pow(2.0, -A_zeta + 1)*A_zeta*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*((-i_x + j_x)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik) + (-i_x + k_x)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)) + (2*i_x - j_x - k_x)/(dij*dik))*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_x - j_x)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_x - k_x)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);
                                    NumericType accum_i_y = -pow(2.0, -A_zeta + 1)*A_eta*((i_y - j_y)/dij + (i_y - k_y)/dik)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - pow(2.0, -A_zeta + 1)*A_zeta*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*((-i_y + j_y)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik) + (-i_y + k_y)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)) + (2*i_y - j_y - k_y)/(dij*dik))*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_y - j_y)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_y - k_y)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);
                                    NumericType accum_i_z = -pow(2.0, -A_zeta + 1)*A_eta*((i_z - j_z)/dij + (i_z - k_z)/dik)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - pow(2.0, -A_zeta + 1)*A_zeta*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*((-i_z + j_z)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik) + (-i_z + k_z)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)) + (2*i_z - j_z - k_z)/(dij*dik))*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_z - j_z)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_z - k_z)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);

                                    NumericType accumulant = accum_i_x * X_grads[g_atomic_num_i] + accum_i_y * Y_grads[g_atomic_num_i] + accum_i_z * Z_grads[g_atomic_num_i];
                                    angular_feature_buffer_i[linearize(an_j, an_k, t, s)] += accumulant;

                                    NumericType accum_j_x = -pow(2.0, -A_zeta + 1)*A_eta*(-i_x + j_x)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - pow(2.0, -A_zeta + 1)*A_zeta*((-i_x + k_x)/(dij*dik) + (i_x - j_x)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_x + j_x)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij);
                                    NumericType accum_j_y = -pow(2.0, -A_zeta + 1)*A_eta*(-i_y + j_y)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - pow(2.0, -A_zeta + 1)*A_zeta*((-i_y + k_y)/(dij*dik) + (i_y - j_y)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_y + j_y)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij);
                                    NumericType accum_j_z = -pow(2.0, -A_zeta + 1)*A_eta*(-i_z + j_z)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - pow(2.0, -A_zeta + 1)*A_zeta*((-i_z + k_z)/(dij*dik) + (i_z - j_z)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_z + j_z)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij); 

                                    accumulant = accum_j_x * X_grads[g_atomic_num_i] + accum_j_y * Y_grads[g_atomic_num_i] + accum_j_z * Z_grads[g_atomic_num_i];
                                    angular_feature_buffer_i[linearize(an_j, an_k, t, s)] += accumulant;

                                    NumericType accum_k_x = -pow(2.0, -A_zeta + 1)*A_eta*(-i_x + k_x)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - pow(2.0, -A_zeta + 1)*A_zeta*((-i_x + j_x)/(dij*dik) + (i_x - k_x)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_x + k_x)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);
                                    NumericType accum_k_y = -pow(2.0, -A_zeta + 1)*A_eta*(-i_y + k_y)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - pow(2.0, -A_zeta + 1)*A_zeta*((-i_y + j_y)/(dij*dik) + (i_y - k_y)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_y + k_y)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);                                    
                                    NumericType accum_k_z = -pow(2.0, -A_zeta + 1)*A_eta*(-i_z + k_z)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - pow(2.0, -A_zeta + 1)*A_zeta*((-i_z + j_z)/(dij*dik) + (i_z - k_z)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_z + k_z)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);

                                    accumulant = accum_k_x * X_grads[g_atomic_num_i] + accum_k_y * Y_grads[g_atomic_num_i] + accum_k_z * Z_grads[g_atomic_num_i];
                                    angular_feature_buffer_i[linearize(an_j, an_k, t, s)] += accumulant;
                                }
                            }     
                        }
                    }
                }
            }
        }
    }
}

template void featurize_grad_inverse(
    const float *input_Xs,
    const float *input_Ys,
    const float *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols, // denotes where the atom is being displaced to
    const int *scatter_idxs, // used to retrieve the grad multiplication factor for backprop
    const float *X_grads,
    const float *Y_grads,
    const float *Z_grads, 

    float *H_grads,
    float *C_grads,
    float *N_grads,
    float *O_grads);

template void featurize_grad_inverse(
    const double *input_Xs,
    const double *input_Ys,
    const double *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols, // denotes where the atom is being displaced to
    const int *scatter_idxs, // used to retrieve the grad multiplication factor for backprop
    const double *X_grads,
    const double *Y_grads,
    const double *Z_grads, 

    double *H_grads,
    double *C_grads,
    double *N_grads,
    double *O_grads);

template<typename NumericType>
void featurize_grad_cpu(
    const NumericType *input_Xs,
    const NumericType *input_Ys,
    const NumericType *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols, // denotes where the atom is being displaced to
    const int *scatter_idxs, // used to retrieve the grad multiplication factor for backprop
    const NumericType *H_grads,
    const NumericType *C_grads,
    const NumericType *N_grads,
    const NumericType *O_grads,
    NumericType *X_grads,
    NumericType *Y_grads,
    NumericType *Z_grads
    ) {
  
    // this is a loop over all unique pairs and triples
    for(int mol_idx=0; mol_idx < n_mols; mol_idx++) {

        int num_atoms = input_MACs[mol_idx];

        for(int i_idx = 0; i_idx < num_atoms; i_idx++) {

            int g_atom_idx_i = mol_offsets[mol_idx] + i_idx;
            int g_atomic_num_i = input_As[g_atom_idx_i];

            NumericType i_x = input_Xs[g_atom_idx_i];
            NumericType i_y = input_Ys[g_atom_idx_i];
            NumericType i_z = input_Zs[g_atom_idx_i];

            NumericType *X_grads_i, *Y_grads_i, *Z_grads_i;

            const NumericType *output_buffer_i;
            if(g_atomic_num_i == 0) {
                output_buffer_i = H_grads;
            } else if(g_atomic_num_i == 1) {
                output_buffer_i = C_grads;
            } else if(g_atomic_num_i == 2) {
                output_buffer_i = N_grads;
            } else {
                output_buffer_i = O_grads;
            }

            const NumericType *radial_feature_buffer_i = output_buffer_i + scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + 0;
            const NumericType *angular_feature_buffer_i = output_buffer_i + scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + RADIAL_FEATURE_SIZE;

            for(int j_idx = 0; j_idx < num_atoms; j_idx++) {
                int g_atom_idx_j = mol_offsets[mol_idx]+j_idx;
                int g_atomic_num_j = input_As[g_atom_idx_j];

                NumericType j_x = input_Xs[g_atom_idx_j];
                NumericType j_y = input_Ys[g_atom_idx_j];
                NumericType j_z = input_Zs[g_atom_idx_j];

                NumericType d_ij_x = i_x - j_x;
                NumericType d_ij_y = i_y - j_y;
                NumericType d_ij_z = i_z - j_z;

                NumericType r_ij = dist_diff(d_ij_x, d_ij_y, d_ij_z);

                // note: the potential is computed as i_idx < j_idx but it NumericType counts.
                if(r_ij < R_Rc && i_idx != j_idx) {
                    for(int r_idx = 0; r_idx < NUM_R_Rs; r_idx++) {
                        NumericType d_y_i = radial_feature_buffer_i[input_As[g_atom_idx_j] * NUM_R_Rs + r_idx]; // gradient broadcast

                        NumericType accum_i_x = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(i_x - j_x)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(i_x - j_x)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);
                        NumericType accum_i_y = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(i_y - j_y)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(i_y - j_y)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);
                        NumericType accum_i_z = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(i_z - j_z)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(i_z - j_z)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);

                        X_grads[g_atom_idx_i] += accum_i_x * d_y_i;
                        Y_grads[g_atom_idx_i] += accum_i_y * d_y_i;
                        Z_grads[g_atom_idx_i] += accum_i_z * d_y_i;

                        NumericType accum_j_x = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(-i_x + j_x)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(-i_x + j_x)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);
                        NumericType accum_j_y = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(-i_y + j_y)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(-i_y + j_y)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);
                        NumericType accum_j_z = -2*R_eta*(-R_Rs[r_idx] + r_ij)*(-i_z + j_z)*(0.5*cos(M_PI*r_ij/R_Rc) + 0.5)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))/r_ij - 0.5*M_PI*(-i_z + j_z)*exp(-R_eta*pow(-R_Rs[r_idx] + r_ij, 2))*sin(M_PI*r_ij/R_Rc)/(R_Rc*r_ij);

                        X_grads[g_atom_idx_j] += accum_j_x * d_y_i;
                        Y_grads[g_atom_idx_j] += accum_j_y * d_y_i;
                        Z_grads[g_atom_idx_j] += accum_j_z * d_y_i;

                    }
                }

                if(r_ij < A_Rc) {
                    for(int k_idx = j_idx+1; k_idx < num_atoms; k_idx++) {
                        if(i_idx == j_idx || i_idx == k_idx || j_idx == k_idx) {
                            continue;
                        }
                        int g_atom_idx_k = mol_offsets[mol_idx]+k_idx;
                        int g_atomic_num_k = input_As[g_atom_idx_k];

                        const int an_j = input_As[g_atom_idx_j];
                        const int an_k = input_As[g_atom_idx_k];

                        NumericType k_x = input_Xs[g_atom_idx_k];
                        NumericType k_y = input_Ys[g_atom_idx_k];
                        NumericType k_z = input_Zs[g_atom_idx_k];

                        NumericType d_ik_x = i_x - k_x;
                        NumericType d_ik_y = i_y - k_y;
                        NumericType d_ik_z = i_z - k_z;

                        NumericType r_ik = dist_diff(d_ik_x, d_ik_y, d_ik_z);

                        if(r_ik < A_Rc) {
                            for(int t=0; t < NUM_A_THETAS; t++) {
                                for(int s=0; s < NUM_A_RS; s++) {

                                    NumericType d_y_i = angular_feature_buffer_i[linearize(an_j, an_k, t, s)];   

                                    NumericType dij = sqrt(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2));
                                    NumericType dik = sqrt(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2));
                                    NumericType ijk_swizzle = (i_x - j_x)*(i_x - k_x) + (i_y - j_y)*(i_y - k_y) + (i_z - j_z)*(i_z - k_z);

                                    NumericType accum_i_x = -pow(2.0, -A_zeta + 1)*A_eta*((i_x - j_x)/dij + (i_x - k_x)/dik)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - pow(2.0, -A_zeta + 1)*A_zeta*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*((-i_x + j_x)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik) + (-i_x + k_x)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)) + (2*i_x - j_x - k_x)/(dij*dik))*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_x - j_x)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_x - k_x)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);
                                    NumericType accum_i_y = -pow(2.0, -A_zeta + 1)*A_eta*((i_y - j_y)/dij + (i_y - k_y)/dik)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - pow(2.0, -A_zeta + 1)*A_zeta*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*((-i_y + j_y)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik) + (-i_y + k_y)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)) + (2*i_y - j_y - k_y)/(dij*dik))*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_y - j_y)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_y - k_y)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);
                                    NumericType accum_i_z = -pow(2.0, -A_zeta + 1)*A_eta*((i_z - j_z)/dij + (i_z - k_z)/dik)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2)) - pow(2.0, -A_zeta + 1)*A_zeta*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*((-i_z + j_z)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik) + (-i_z + k_z)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)) + (2*i_z - j_z - k_z)/(dij*dik))*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_z - j_z)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(i_z - k_z)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);

                                    X_grads[g_atom_idx_i] += accum_i_x * d_y_i;
                                    Y_grads[g_atom_idx_i] += accum_i_y * d_y_i;
                                    Z_grads[g_atom_idx_i] += accum_i_z * d_y_i;

                                    NumericType accum_j_x = -pow(2.0, -A_zeta + 1)*A_eta*(-i_x + j_x)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - pow(2.0, -A_zeta + 1)*A_zeta*((-i_x + k_x)/(dij*dik) + (i_x - j_x)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_x + j_x)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij);
                                    NumericType accum_j_y = -pow(2.0, -A_zeta + 1)*A_eta*(-i_y + j_y)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - pow(2.0, -A_zeta + 1)*A_zeta*((-i_y + k_y)/(dij*dik) + (i_y - j_y)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_y + j_y)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij);
                                    NumericType accum_j_z = -pow(2.0, -A_zeta + 1)*A_eta*(-i_z + j_z)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dij - pow(2.0, -A_zeta + 1)*A_zeta*((-i_z + k_z)/(dij*dik) + (i_z - j_z)*(ijk_swizzle)/(pow(pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2), 1.5)*dik))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_z + j_z)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dij/A_Rc)/(A_Rc*dij); 

                                    X_grads[g_atom_idx_j] += accum_j_x * d_y_i;
                                    Y_grads[g_atom_idx_j] += accum_j_y * d_y_i;
                                    Z_grads[g_atom_idx_j] += accum_j_z * d_y_i;

                                    NumericType accum_k_x = -pow(2.0, -A_zeta + 1)*A_eta*(-i_x + k_x)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - pow(2.0, -A_zeta + 1)*A_zeta*((-i_x + j_x)/(dij*dik) + (i_x - k_x)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_x + k_x)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);
                                    NumericType accum_k_y = -pow(2.0, -A_zeta + 1)*A_eta*(-i_y + k_y)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - pow(2.0, -A_zeta + 1)*A_zeta*((-i_y + j_y)/(dij*dik) + (i_y - k_y)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_y + k_y)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);
                                    NumericType accum_k_z = -pow(2.0, -A_zeta + 1)*A_eta*(-i_z + k_z)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*(-A_Rs[s] + (0.5)*dij + (0.5)*dik)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))/dik - pow(2.0, -A_zeta + 1)*A_zeta*((-i_z + j_z)/(dij*dik) + (i_z - k_z)*(ijk_swizzle)/(dij*pow(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2), 1.5)))*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*(0.5*cos(M_PI*dik/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(A_thetas[t] - acos((ijk_swizzle)/(dij*dik)))/(sqrt(-pow(ijk_swizzle, 2)/((pow(i_x - j_x, 2) + pow(i_y - j_y, 2) + pow(i_z - j_z, 2))*(pow(i_x - k_x, 2) + pow(i_y - k_y, 2) + pow(i_z - k_z, 2))) + 1)*(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1)) - 0.5*pow(2.0, -A_zeta + 1)*M_PI*(-i_z + k_z)*(0.5*cos(M_PI*dij/A_Rc) + 0.5)*pow(cos(A_thetas[t] - acos((ijk_swizzle)/(dij*dik))) + 1, A_zeta)*exp(-A_eta*pow(-A_Rs[s] + (0.5)*dij + (0.5)*dik, 2))*sin(M_PI*dik/A_Rc)/(A_Rc*dik);

                                    X_grads[g_atom_idx_k] += accum_k_x * d_y_i;
                                    Y_grads[g_atom_idx_k] += accum_k_y * d_y_i;
                                    Z_grads[g_atom_idx_k] += accum_k_z * d_y_i;
                                }
                            }     
                        }
                    }
                }
            }
        }
    }
}

// instantiation
template void featurize_grad_cpu(
    const double *input_Xs,
    const double *input_Ys,
    const double *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols, // denotes where the atom is being displaced to
    const int *scatter_idxs, // used to retrieve the grad multiplication factor for backprop
    const double *H_grads,
    const double *C_grads,
    const double *N_grads,
    const double *O_grads,
    double *X_grads,
    double *Y_grads,
    double *Z_grads
);

template void featurize_grad_cpu(
    const float *input_Xs,
    const float *input_Ys,
    const float *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols, // denotes where the atom is being displaced to
    const int *scatter_idxs, // used to retrieve the grad multiplication factor for backprop
    const float *H_grads,
    const float *C_grads,
    const float *N_grads,
    const float *O_grads,
    float *X_grads,
    float *Y_grads,
    float *Z_grads
);


template<typename NumericType>
void featurize_cpu(
    const NumericType *input_Xs,
    const NumericType *input_Ys,
    const NumericType *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols,
    const int *scatter_idxs, // denotes where the atom is being displaced to
    NumericType *X_feat_out_H,
    NumericType *X_feat_out_C,
    NumericType *X_feat_out_N,
    NumericType *X_feat_out_O) {
 
    // std::cout << "start featurize_cpu" << std::endl;
    // std::cout << type_name<decltype(input_X)>() << std::endl;

    // this is a loop over all unique pairs and triples
    for(int mol_idx=0; mol_idx < n_mols; mol_idx++) {

        int num_atoms = input_MACs[mol_idx];

        for(int i_idx = 0; i_idx < num_atoms; i_idx++) {

            int g_atom_idx_i = mol_offsets[mol_idx] + i_idx;
            int g_atomic_num_i = input_As[g_atom_idx_i];

            NumericType i_x = input_Xs[g_atom_idx_i];
            NumericType i_y = input_Ys[g_atom_idx_i];
            NumericType i_z = input_Zs[g_atom_idx_i];

            NumericType *X_feat_out_i;
            if(g_atomic_num_i == 0) {
                X_feat_out_i = X_feat_out_H;
            } else if(g_atomic_num_i == 1) {
                X_feat_out_i = X_feat_out_C;
            } else if(g_atomic_num_i == 2) {
                X_feat_out_i = X_feat_out_N;
            } else {
                X_feat_out_i = X_feat_out_O;
            }

            NumericType *radial_feature_buffer_i = X_feat_out_i + scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + 0;
            NumericType *angular_feature_buffer_i = X_feat_out_i + scatter_idxs[g_atom_idx_i]*TOTAL_FEATURE_SIZE + RADIAL_FEATURE_SIZE;

            for(int j_idx = 0; j_idx < num_atoms; j_idx++) {
                int g_atom_idx_j = mol_offsets[mol_idx]+j_idx;
                int g_atomic_num_j = input_As[g_atom_idx_j];

                NumericType j_x = input_Xs[g_atom_idx_j];
                NumericType j_y = input_Ys[g_atom_idx_j];
                NumericType j_z = input_Zs[g_atom_idx_j];

                NumericType d_ij_x = i_x - j_x;
                NumericType d_ij_y = i_y - j_y;
                NumericType d_ij_z = i_z - j_z;

                NumericType r_ij = dist_diff(d_ij_x, d_ij_y, d_ij_z);

                NumericType *X_feat_out_j;
                if(g_atomic_num_j == 0) {
                    X_feat_out_j = X_feat_out_H;
                } else if(g_atomic_num_j == 1) {
                    X_feat_out_j = X_feat_out_C;
                } else if(g_atomic_num_j == 2) {
                    X_feat_out_j = X_feat_out_N;
                } else {
                    X_feat_out_j = X_feat_out_O;
                }

                NumericType *radial_feature_buffer_j = X_feat_out_j + scatter_idxs[g_atom_idx_j]*TOTAL_FEATURE_SIZE + 0;

                if(r_ij < R_Rc && i_idx < j_idx) {
                    for(int r_idx = 0; r_idx < NUM_R_Rs; r_idx++) {
                        NumericType fC = f_C(r_ij, R_Rc);
                        NumericType lhs = exp(-R_eta * pow(r_ij - R_Rs[r_idx], 2.0));
                        NumericType summand = lhs * fC;
                        radial_feature_buffer_i[input_As[g_atom_idx_j] * NUM_R_Rs + r_idx] += summand;
                        radial_feature_buffer_j[input_As[g_atom_idx_i] * NUM_R_Rs + r_idx] += summand;
                    }
                }

                NumericType A_f_C_ij = f_C(r_ij, A_Rc);

                if(r_ij < A_Rc) {
                    for(int k_idx = j_idx+1; k_idx < num_atoms; k_idx++) {
                        if(i_idx == j_idx || i_idx == k_idx || j_idx == k_idx) {
                            continue;
                        }
                        int g_atom_idx_k = mol_offsets[mol_idx]+k_idx;

                        const int an_j = input_As[g_atom_idx_j];
                        const int an_k = input_As[g_atom_idx_k];

                        NumericType k_x = input_Xs[g_atom_idx_k];
                        NumericType k_y = input_Ys[g_atom_idx_k];
                        NumericType k_z = input_Zs[g_atom_idx_k];

                        NumericType d_ik_x = i_x - k_x;
                        NumericType d_ik_y = i_y - k_y;
                        NumericType d_ik_z = i_z - k_z;

                        NumericType r_ik = dist_diff(d_ik_x, d_ik_y, d_ik_z);

                        if(r_ik < A_Rc) {

                            // TODO(YTZ): replace with arctan2 trick
                            NumericType inner = (d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z) / (r_ij * r_ik);
                            inner = fmax(inner, -1.0);
                            inner = fmin(inner,  1.0);
                            NumericType theta_ijk = acos(inner);
                            // super useful debug
                            // if(isnan(theta_ijk) || isinf(theta_ijk)) {
                            //     printf("WTF NAN/INF: %d, %d, %d, r_ij, r_ik, %f, %f, top %f, bottom %f, i_coords:(%f, %f, %f), j_coords(%f, %f, %f), k_coords(%f, %f, %f)\n",
                            //         g_atom_idx_i, g_atom_idx_j, g_atom_idx_k, r_ij, r_ik, d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z, r_ij * r_ik, i_x, i_y, i_z, j_x, j_y, j_z, k_x, k_y, k_z);
                            // }

                            NumericType A_f_C_ik = f_C(r_ik, A_Rc);
                            for(int t=0; t < NUM_A_THETAS; t++) {
                                for(int s=0; s < NUM_A_RS; s++) {
                                    NumericType summand = pow(2.0, 1-A_zeta) * pow(1+cos(theta_ijk - A_thetas[t]), A_zeta) * exp(-A_eta*pow((r_ij + r_ik)/2 - A_Rs[s], 2)) * A_f_C_ij * A_f_C_ik;
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

// instantiation
template void featurize_cpu(
    const double *input_Xs,
    const double *input_Ys,
    const double *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols,
    const int *scatter_idxs, // denotes where the atom is being displaced to
    double *X_feat_out_H,
    double *X_feat_out_C,
    double *X_feat_out_N,
    double *X_feat_out_O);

// instantiation
template void featurize_cpu(
    const float *input_Xs,
    const float *input_Ys,
    const float *input_Zs,
    const int *input_As,
    const int *mol_offsets,
    const int *input_MACs,
    const int n_mols,
    const int *scatter_idxs, // denotes where the atom is being displaced to
    float *X_feat_out_H,
    float *X_feat_out_C,
    float *X_feat_out_N,
    float *X_feat_out_O);