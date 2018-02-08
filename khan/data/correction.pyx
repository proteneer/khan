# cython: profile=True
# cython: linetrace=True

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
c_self_terms[0] = -3.61495025e+02
c_self_terms[1] = -2.38566440e+04
c_self_terms[2] = -3.43234157e+04
c_self_terms[3] = -4.71784651e+04

cdef double c_pair_terms[4][4][4]

c_pair_terms[0][1] = [-1.35927769e+02, 2.10919757e+03, -6.80491343e+03, 5.88608669e+03]
c_pair_terms[0][2] = [-1.74835391e+02, 2.79778489e+03, -8.27595549e+03, 6.60117401e+03]
c_pair_terms[0][3] = [-2.66558100e+02, 3.47283119e+03, -9.40328190e+03, 7.06803912e+03]
c_pair_terms[1][1] = [ 2.36476049e+01, 3.22775414e+02, -3.46198683e+03, 5.16847643e+03]
c_pair_terms[1][2] = [ 1.00037527e+02, 4.65919734e+02, -4.85364601e+03, 6.35979202e+03]
c_pair_terms[1][3] = [ 1.27547041e+01, 2.06357637e+03, -1.00000000e+04, 1.11347193e+04]
c_pair_terms[2][2] = [ 1.72507376e+02, 1.51680516e+03, -9.99999939e+03, 1.26617984e+04]
c_pair_terms[2][3] = [-7.21578715e+01, 2.66909212e+03, -9.99934500e+03, 1.03047549e+04]
c_pair_terms[3][3] = [-2.77910695e+02, 3.29117774e+03, -1.37665219e+03, 1.68165667e+02]

c_pair_terms[1][0] = [-1.35927769e+02, 2.10919757e+03, -6.80491343e+03, 5.88608669e+03]
c_pair_terms[2][0] = [-1.74835391e+02, 2.79778489e+03, -8.27595549e+03, 6.60117401e+03]
c_pair_terms[3][0] = [-2.66558100e+02, 3.47283119e+03, -9.40328190e+03, 7.06803912e+03]
# c_pair_terms[1][1] = [ 2.36476049e+01, 3.22775414e+02, -3.46198683e+03, 5.16847643e+03], dtyp
c_pair_terms[2][1] = [ 1.00037527e+02, 4.65919734e+02, -4.85364601e+03, 6.35979202e+03]
c_pair_terms[3][1] = [ 1.27547041e+01, 2.06357637e+03, -1.00000000e+04, 1.11347193e+04]
# c_pair_terms[2][2] = [ 1.72507376e+02, 1.51680516e+03, -9.99999939e+03, 1.26617984e+04], dtyp
c_pair_terms[3][2] = [-7.21578715e+01, 2.66909212e+03, -9.99934500e+03, 1.03047549e+04]


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
        # printf("i1: %i\n", i)
        inner_i_A = Z_ptr[i]
        i_x = X_ptr[i*3+0]
        i_y = X_ptr[i*3+1]
        i_z = X_ptr[i*3+2]
        for j in range(i+1, num_atoms):
            inner_j_A = Z_ptr[j]

            if inner_i_A == 0 and inner_j_A == 0:
                pass
            else:
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