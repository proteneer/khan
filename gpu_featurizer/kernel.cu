#include <iostream>
#include <vector>
#include <chrono>

#include "cnpy.h" // utility for reading numpy .npy files.

/*

YTZ Notes:

This kernel implements the ANI-1 featurization scheme. It can process about 3.6 million samples/minute
*/

const int MAX_ATOM_TYPES = 4;

const int NUM_R_Rs = 16;
const int RADIAL_FEATURE_SIZE = MAX_ATOM_TYPES * NUM_R_Rs;

const float R_eta = 16;
const float R_Rc = 4.6;

const float A_Rc = 3.1;
const float A_eta = 6.0;
const float A_zeta = 8.0;
const int NUM_A_THETAS = 8;
const int NUM_A_RS = 4;

const int ANGULAR_FEATURE_SIZE = NUM_A_RS * NUM_A_THETAS * (MAX_ATOM_TYPES * (MAX_ATOM_TYPES+1) / 2);

const int TOTAL_FEATURE_SIZE = RADIAL_FEATURE_SIZE + ANGULAR_FEATURE_SIZE;

__device__ const float R_Rs[NUM_R_Rs] = {
    5.0000000e-01,
    7.5625000e-01,
    1.0125000e+00,
    1.2687500e+00,
    1.5250000e+00,
    1.7812500e+00,
    2.0375000e+00,
    2.2937500e+00,
    2.5500000e+00,
    2.8062500e+00,
    3.0625000e+00,
    3.3187500e+00,
    3.5750000e+00,
    3.8312500e+00,
    4.0875000e+00,
    4.3437500e+00
};

__device__ const float A_thetas[NUM_A_THETAS] = {
    0.0000000e+00,
    7.8539816e-01,
    1.5707963e+00,
    2.3561945e+00,
    3.1415927e+00,
    3.9269908e+00,
    4.7123890e+00,
    5.4977871e+00
};

__device__ const float A_Rs[NUM_A_RS] = {
    5.0000000e-01,
    1.1500000e+00,
    1.8000000e+00,
    2.4500000e+00,
};


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

// Remind yutong to document what these pointers are.
__global__ void featurize(
    const float *Xs,
    const float *Ys,
    const float *Zs,

    const int *atomic_nums,
    const int *mol_offsets,
    const int *mol_atom_count,
    const int num_mols, // actually equal to blockDim.x

    float *X_feat_out) {

    int mol_idx = blockIdx.x;
    int local_atom_idx = threadIdx.x; // local_local_atom_idx
    int num_atoms = mol_atom_count[blockIdx.x];


    // printf("num_atoms %d \n", num_atoms);

    if (local_atom_idx >= num_atoms) {
        return;
    }

    // todo: cache into shared mem
    // load all the x y z coordinates
    int g_atom_idx_i = mol_offsets[mol_idx]+local_atom_idx;

    float i_x = Xs[g_atom_idx_i];
    float i_y = Ys[g_atom_idx_i];
    float i_z = Zs[g_atom_idx_i];


    // printf("%d %d %d (%f, %f, %f)\n", mol_idx, local_atom_idx, num_atoms, i_x, i_y, i_z);

    float *radial_feature_buffer_i = X_feat_out + g_atom_idx_i*TOTAL_FEATURE_SIZE + 0;
    float *angular_feature_buffer_i = X_feat_out + g_atom_idx_i*TOTAL_FEATURE_SIZE + RADIAL_FEATURE_SIZE;

    for(int j=0; j < num_atoms; j++) {

        int g_atom_idx_j = mol_offsets[mol_idx]+j;

        float j_x = Xs[g_atom_idx_j];
        float j_y = Ys[g_atom_idx_j];
        float j_z = Zs[g_atom_idx_j];

        float d_ij_x = i_x - j_x;
        float d_ij_y = i_y - j_y;
        float d_ij_z = i_z - j_z;

        // printf("(%f, %f, %f)\n", d_ij_x, d_ij_y, d_ij_z);

        float r_ij = dist_diff(d_ij_x, d_ij_y, d_ij_z);

        float *radial_feature_buffer_j = X_feat_out + g_atom_idx_j*TOTAL_FEATURE_SIZE + 0;
        // float *angular_feature_buffer_j = X_feat_out + g_atom_idx_j*TOTAL_FEATURE_SIZE + RADIAL_FEATURE_SIZE;


        // radial features
        if(r_ij < R_Rc and local_atom_idx < j) {
            for(int r_idx = 0; r_idx < NUM_R_Rs; r_idx++) {
                float summand = expf(-R_eta * powf(r_ij - R_Rs[r_idx], 2.0)) * f_C(r_ij, R_Rc);

                // exploit symmetry of the atomic adds
                atomicAdd(radial_feature_buffer_i + atomic_nums[g_atom_idx_j] * NUM_R_Rs + r_idx, summand);
                atomicAdd(radial_feature_buffer_j + atomic_nums[g_atom_idx_i] * NUM_R_Rs + r_idx, summand);
            }
        }

        float A_f_C_ij = f_C(r_ij, A_Rc);

        if(r_ij < A_Rc) {

            for(size_t k=0; k < num_atoms; k++) {

                if(local_atom_idx == j || local_atom_idx == k || j == k) {
                    continue;
                }

                const int an_i = atomic_nums[local_atom_idx];
                const int an_j = atomic_nums[j];

                int g_atom_idx_k = mol_offsets[mol_idx]+k;

                float k_x = Xs[g_atom_idx_k];
                float k_y = Ys[g_atom_idx_k];
                float k_z = Zs[g_atom_idx_k];

                float d_ik_x = i_x - k_x;
                float d_ik_y = i_y - k_y;
                float d_ik_z = i_z - k_z;

                float r_ik = dist_diff(d_ik_x, d_ik_y, d_ik_z);

                if(r_ik < A_Rc) {
                    float theta_ijk = acos((d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z) / (r_ij * r_ik));
                    // printf("%d t_ijk: %f\n", local_atom_idx, theta_ijk);
                    float A_f_C_ik = f_C(r_ik, A_Rc);
                    for(int t=0; t < NUM_A_THETAS; t++) {
                        for(int s=0; s < NUM_A_RS; s++) {
                            float summand = 2*(1-A_zeta) * powf(1+cosf(theta_ijk - A_thetas[t]), A_zeta) * expf(-A_eta*powf((r_ij + r_ik)/2 - A_Rs[s], 2)) * A_f_C_ij * A_f_C_ik;
                            atomicAdd(angular_feature_buffer_i + linearize(an_i, an_j, t, s), summand);
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

int main(void) {

    auto X_obj = cnpy::npy_load("Xs.npy");
    auto Y_obj = cnpy::npy_load("Ys.npy");
    auto Z_obj = cnpy::npy_load("Zs.npy");
    auto A_obj = cnpy::npy_load("As.npy");

    auto MOs = cnpy::npy_load("MOs.npy");
    auto MACs = cnpy::npy_load("MACs.npy");

    float *d_Xs = cudaMallocSimple<float>(X_obj.shape[0]);
    float *d_Ys = cudaMallocSimple<float>(Y_obj.shape[0]);
    float *d_Zs = cudaMallocSimple<float>(Z_obj.shape[0]);
    int *d_As = cudaMallocSimple<int>(A_obj.shape[0]);
    int *d_MOs = cudaMallocSimple<int>(MOs.shape[0]);
    int *d_MACs = cudaMallocSimple<int>(MACs.shape[0]); // max

    size_t n_atoms = X_obj.shape[0];
    size_t n_mols  = MOs.shape[0];

    float* d_X_feat;

    printf("Total number of atoms: %d \n", n_atoms);

    std::cout << "mallocing:" << n_atoms*TOTAL_FEATURE_SIZE*sizeof(float) << "bytes\n";

    cudaMalloc(&d_X_feat, n_atoms*TOTAL_FEATURE_SIZE*sizeof(float)); 

    auto start = Clock::now();

    for(size_t i=0; i < 100; i++) {
        std::cout << i << std::endl;    
        cudaCopySimple(X_obj.data<float>(), X_obj.shape[0], d_Xs);
        cudaCopySimple(Y_obj.data<float>(), Y_obj.shape[0], d_Ys);
        cudaCopySimple(Z_obj.data<float>(), Z_obj.shape[0], d_Zs);
        cudaCopySimple(A_obj.data<int>(), A_obj.shape[0], d_As);

        cudaCopySimple(MOs.data<int>(), MOs.shape[0], d_MOs);
        cudaCopySimple(MACs.data<int>(), MACs.shape[0], d_MACs); // max

        assert(cudaMemset(d_X_feat, 0, n_atoms*TOTAL_FEATURE_SIZE*sizeof(float)) == 0);

        std::cout << "n_mols" << n_mols << std::endl;

        featurize<<<n_mols, 32>>>(
            d_Xs,
            d_Ys,
            d_Zs,
            d_As,
            d_MOs,
            d_MACs,
            n_mols,
            d_X_feat);


        gpuErrchk( cudaPeekAtLastError() );

        auto end = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();


        std::cout << "DURATION:" << duration << std::endl;

        std::cout << "samples per minute:" << (float(i*n_mols) / duration) * 60 * 1e9 << std::endl;

    }

    std::vector<float> X_feat(n_atoms*TOTAL_FEATURE_SIZE, 0);
    cudaMemcpy(&X_feat[0], d_X_feat, n_atoms*TOTAL_FEATURE_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

}