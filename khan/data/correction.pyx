import numpy as np

cimport numpy as np
cimport cython

from cython.parallel import parallel, prange
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libc.math cimport exp
from libc.math cimport pow

DTYPE_f32 = np.float32
DTYPE_i32 = np.int32

ctypedef np.float32_t DTYPE_t_f32
ctypedef np.int32_t DTYPE_t_i32

cdef double c_self_terms[4]
cdef double c_pair_terms[4][4][4]

c_self_terms[0] = -346.518062
c_self_terms[1] = -23880.556900
c_self_terms[2] = -34338.588200
c_self_terms[3] = -47195.619400
c_pair_terms[0][0] = [-101.847904, 822.168893, -1745.079680, 1561.952700]
c_pair_terms[0][1] = [-59.230037, 1561.106670, -5751.412120, 5201.669380]
c_pair_terms[1][0] = [-59.230037, 1561.106670, -5751.412120, 5201.669380]
c_pair_terms[0][2] = [-109.930509, 2351.606460, -7575.123670, 6256.223940]
c_pair_terms[2][0] = [-109.930509, 2351.606460, -7575.123670, 6256.223940]
c_pair_terms[0][3] = [-233.766348, 3255.726320, -9258.812390, 7157.068490]
c_pair_terms[3][0] = [-233.766348, 3255.726320, -9258.812390, 7157.068490]
c_pair_terms[1][1] = [-218.593551, 2716.999340, -9998.642800, 11011.991300]
c_pair_terms[1][2] = [-106.576244, 2170.245760, -8856.944230, 9518.800200]
c_pair_terms[2][1] = [-106.576244, 2170.245760, -8856.944230, 9518.800200]
c_pair_terms[1][3] = [-75.006032, 2408.505910, -10000.000000, 10705.229300]
c_pair_terms[3][1] = [-75.006032, 2408.505910, -10000.000000, 10705.229300]
c_pair_terms[2][2] = [-17.828686, 2340.157690, -9992.230320, 10620.368400]
c_pair_terms[2][3] = [-98.885354, 2688.195990, -9993.419310, 10591.472300]
c_pair_terms[3][2] = [-98.885354, 2688.195990, -9993.419310, 10591.472300]
c_pair_terms[3][3] = [-111.653267, -355.339147, 10000.000000, 11442.509100]


ctypedef vector[float] float_vec                                                        

@cython.boundscheck(False)
@cython.wraparound(False)
def jamesPairwiseCorrection_C(
    np.ndarray[DTYPE_t_f32, ndim=2, mode="c"] X,
    np.ndarray[DTYPE_t_i32, ndim=1, mode="c"] Z):

    # note that raw point indexing is actually faster than using memoryviews
    cdef float *X_ptr = <float*> X.data
    cdef int *Z_ptr = <int*> Z.data

    cdef int num_atoms = Z.shape[0]

    cdef int inner_i_A = 0
    cdef int inner_j_A = 0

    cdef int i, j

    cdef float i_x, i_y, i_z, j_x, j_y, j_z
    cdef float r2, expr
    cdef float b1 = 0
    cdef float b2 = 0 
    cdef float b3 = 0 
    cdef float b4 = 0 

    cdef int condensed_idx = 0

    cdef double E_self = 0
    cdef double E_pair = 0
    cdef float energy = 0

    for i in range(num_atoms):
        inner_i_A = Z_ptr[i]
        i_x = X_ptr[i*3+0]
        i_y = X_ptr[i*3+1]
        i_z = X_ptr[i*3+2]
        for j in range(i+1, num_atoms):
            inner_j_A = Z_ptr[j]

            # if inner_i_A == 0 and inner_j_A == 0:
                # pass
            # else:
            j_x = X_ptr[j*3+0]
            j_y = X_ptr[j*3+1]
            j_z = X_ptr[j*3+2]

            r2 = pow(i_x-j_x, 2) + pow(i_y-j_y, 2) + pow(i_z-j_z, 2)

            b1 = exp(-0.5*r2)
            b2 = b1*b1
            b3 = b2*b1
            b4 = b3*b1

            # condensed_idx = i*num_atoms + j - i*(i+1)/2 - i - 1
            energy = c_pair_terms[inner_i_A][inner_j_A][0]*b1 + \
                    c_pair_terms[inner_i_A][inner_j_A][1]*b2 + \
                    c_pair_terms[inner_i_A][inner_j_A][2]*b3 + \
                    c_pair_terms[inner_i_A][inner_j_A][3]*b4

            E_pair += energy

    for i in range(num_atoms):
        E_self += c_self_terms[Z_ptr[i]]

    return E_self + E_pair


@cython.boundscheck(False)
@cython.wraparound(False)
def generateBasis(
    np.ndarray[DTYPE_t_f32, ndim=2, mode="c"] X,
    np.ndarray[DTYPE_t_i32, ndim=1, mode="c"] Z):

    cdef float *X_ptr = <float*> X.data
    cdef int *Z_ptr = <int*> Z.data

    cdef int num_atoms = Z.shape[0]

    cdef int inner_i_A = 0
    cdef int inner_j_A = 0

    cdef int i, j

    cdef float i_x, i_y, i_z, j_x, j_y, j_z
    cdef float r2, expr
    cdef float b1 = 0
    cdef float b2 = 0 
    cdef float b3 = 0 
    cdef float b4 = 0 

    cdef int condensed_idx = -1

    cdef double E_pair = 0
    cdef float energy = 0

    # E_pair_res = float_vec(num_atoms*(num_atoms-1)/2)

    cdef int n_atom_types = 4
    cdef int n_params = (n_atom_types)*(n_atom_types+1)/2

    bases = float_vec(n_params*4, 0.0) # 10 parameters

    for i in range(num_atoms):
        # printf("i1: %i\n", i)
        inner_i_A = Z_ptr[i]
        i_x = X_ptr[i*3+0]
        i_y = X_ptr[i*3+1]
        i_z = X_ptr[i*3+2]
        for j in range(i+1, num_atoms):
            inner_j_A = Z_ptr[j]

            # if inner_i_A == 0 and inner_j_A == 0:
            #     pass
            # else:
            j_x = X_ptr[j*3+0]
            j_y = X_ptr[j*3+1]
            j_z = X_ptr[j*3+2]

            r2 = pow(i_x-j_x, 2) + pow(i_y-j_y, 2) + pow(i_z-j_z, 2)

            b1 = exp(-0.5*r2)
            b2 = b1*b1
            b3 = b2*b1
            b4 = b3*b1

            b_i = inner_i_A + 1
            b_j = inner_j_A + 1


            if inner_i_A == 0:
                if inner_j_A == 0:
                    condensed_idx = 0
                elif inner_j_A == 1:
                    condensed_idx = 1
                elif inner_j_A == 2:
                    condensed_idx = 2
                elif inner_j_A == 3:
                    condensed_idx = 3
            elif inner_i_A == 1:
                if inner_j_A == 0:
                    condensed_idx = 1
                elif inner_j_A == 1:
                    condensed_idx = 4
                elif inner_j_A == 2:
                    condensed_idx = 5
                elif inner_j_A == 3:
                    condensed_idx = 6
            elif inner_i_A == 2:
                if inner_j_A == 0:
                    condensed_idx = 2
                elif inner_j_A == 1:
                    condensed_idx = 5
                elif inner_j_A == 2:
                    condensed_idx = 7
                elif inner_j_A == 3:
                    condensed_idx = 8
            elif inner_i_A == 3:
                if inner_j_A == 0:
                    condensed_idx = 3
                elif inner_j_A == 1:
                    condensed_idx = 6
                elif inner_j_A == 2:
                    condensed_idx = 8
                elif inner_j_A == 3:
                    condensed_idx = 9

            if condensed_idx == -1:
                print(inner_i_A, inner_j_A, condensed_idx)
                assert 0

            # condensed_idx = b_i*4 + b_j - b_i*(b_i+1)/2 - b_i - 1
            
            # print(inner_i_A, inner_j_A, condensed_idx)
            # print("????", condensed_idx)

            # print(0*n_params + condensed_idx)
            # print(1*n_params + condensed_idx)
            # print(2*n_params + condensed_idx)
            # print(3*n_params + condensed_idx)

            bases[0*n_params + condensed_idx] += b1
            bases[1*n_params + condensed_idx] += b2
            bases[2*n_params + condensed_idx] += b3
            bases[3*n_params + condensed_idx] += b4

    # assert 0

    return bases
