# cython: linetrace=True

FOO = "BAR"


import numpy as np

cimport numpy as np
cimport cython

from cython.parallel import parallel, prange
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libc.math cimport exp, pow, sqrt, cos, acos, M_PI
# from libc.math cimport pow

cdef int MAX_ATOM_TYPES = 4

# Radial Constants
cdef float R_eta = 16.0
cdef float R_Rc = 4.6
cdef float R_Rs[16]
R_Rs = [
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
]



cdef int NUM_R_RS = sizeof(R_Rs) / sizeof(R_Rs[0])
cdef int RADIAL_FEATURE_SIZE = MAX_ATOM_TYPES * NUM_R_RS

# Angular Constants
cdef float A_Rc = 3.1,
cdef float A_eta = 6.0,
cdef float A_zeta = 8.0
cdef float A_thetas[8]
cdef float A_Rs[4]
A_thetas = [
    0.0000000e+00,
    7.8539816e-01,
    1.5707963e+00,
    2.3561945e+00,
    3.1415927e+00,
    3.9269908e+00,
    4.7123890e+00,
    5.4977871e+00]
A_Rs = [
    5.0000000e-01,
    1.1500000e+00,
    1.8000000e+00,
    2.4500000e+00
]


cdef int NUM_A_THETAS = sizeof(A_thetas) / sizeof(A_thetas[0])
cdef int NUM_A_RS = sizeof(A_Rs) / sizeof(A_Rs[0])
cdef int ANGULAR_FEATURE_SIZE = NUM_A_RS * NUM_A_THETAS * (MAX_ATOM_TYPES * (MAX_ATOM_TYPES+1) / 2)
cdef int TOTAL_FEATURE_SIZE = ANGULAR_FEATURE_SIZE + RADIAL_FEATURE_SIZE

@cython.boundscheck(False)
@cython.wraparound(False)
# @cython.binding(True)
cdef void radial_symmetry(
    float* dist_matrix,
    float* feature_buffer,
    size_t n_atoms,
    int* Z_ptr) nogil:

    cdef int i, j, r_f
    cdef float r_ij
    cdef float summand = 0.0


    cdef float *radial_feature_buffer_i;
    cdef float *radial_feature_buffer_j;
    for i in range(n_atoms):
        radial_feature_buffer_i = &feature_buffer[i*TOTAL_FEATURE_SIZE+0]
        for j in range(n_atoms):

            if i == j:
                # skip due to symmetry
                continue

            if i > j:
                # skip since we compute the term once then add contribution in reverse since 
                # rij == rji
                continue

            radial_feature_buffer_j = &feature_buffer[j*TOTAL_FEATURE_SIZE+0]

            r_ij = dist_matrix[i*n_atoms+j]
            if r_ij < R_Rc:
                for r_f in range(NUM_R_RS):
                    summand = exp(-R_eta * pow(r_ij - R_Rs[r_f], 2.0)) * f_C(r_ij, R_Rc)
                    # printf("%d %.4f %.4f \n", Z_ptr[j], exp(-R_eta * pow(r_ij - R_Rs[r_f], 2.0)), f_C(r_ij, R_Rc))
                    radial_feature_buffer_i[Z_ptr[j] * MAX_ATOM_TYPES + r_f] += summand
                    radial_feature_buffer_j[Z_ptr[i] * MAX_ATOM_TYPES + r_f] += summand


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float f_C(float r_ij, float r_c) nogil:
    if r_ij <= r_c:
        return 0.5 * cos((M_PI * r_ij) / r_c) + 0.5
    else:
        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
# @cython.binding(True)
cdef void angular_symmetry(
    float *dist_matrix,
    float *feature_buffer,
    float *X_ptr,
    size_t n_atoms,
    int *Z_ptr) nogil:

    cdef int i, j, k
    cdef int t_s
    cdef int a_f

    cdef float d_ij_x, d_ij_y, d_ij_z
    cdef float d_ik_x, d_ik_y, d_ik_z

    cdef float r_ij
    cdef float r_ik

    cdef float f_C_ij = 0.0
    cdef float f_C_ik = 0.0

    cdef float summand = 0

    cdef float *angular_feature_buffer;

    for i in range(n_atoms):

        angular_feature_buffer = &feature_buffer[i*TOTAL_FEATURE_SIZE+RADIAL_FEATURE_SIZE]

        for j in range(n_atoms):
            r_ij = dist_matrix[i*n_atoms+j]

            d_ij_x = X_ptr[3*i + 0] - X_ptr[3*j + 0]
            d_ij_y = X_ptr[3*i + 1] - X_ptr[3*j + 1]
            d_ij_z = X_ptr[3*i + 2] - X_ptr[3*j + 2]

            if r_ij < A_Rc:
                f_C_ij = f_C(r_ij, A_Rc)
                for k in range(n_atoms):

                    if i == j or i == k or j == k:
                        continue

                    if Z_ptr[j] > Z_ptr[k]:
                        continue

                    r_ik = dist_matrix[i*n_atoms+k]
                    if r_ik < A_Rc:

                        d_ik_x = X_ptr[3*i + 0] - X_ptr[3*k + 0]
                        d_ik_y = X_ptr[3*i + 1] - X_ptr[3*k + 1]
                        d_ik_z = X_ptr[3*i + 2] - X_ptr[3*k + 2]

                        # printf("%f, %f, \n", r_ij, r_ik)

                        theta_ijk = acos((d_ij_x*d_ik_x + d_ij_y*d_ik_y + d_ij_z*d_ik_z) / (r_ij * r_ik))

                        # printf("(%d,%d,%d) %f\n", i,j,k, theta_ijk)

                        f_C_ik = f_C(r_ik, A_Rc)

                        for a_f in range(NUM_A_THETAS):
                            for t_s in range(NUM_A_RS):
                                summand = 2*(1-A_zeta) * pow(1+cos(theta_ijk - A_thetas[t_s]), A_zeta) * exp(-A_eta*pow((r_ij + r_ik)/2 - A_Rs[a_f], 2)) * f_C_ij * f_C_ik
                                angular_feature_buffer[A_map[Z_ptr[j]][Z_ptr[k]][a_f][t_s]] += summand



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ANI1_single(
    float *X_ptr,
    int *Z_ptr,
    float *res,
    int n_atoms) nogil:
    cdef vector[float] dist_matrix = vector[float](n_atoms*n_atoms)

    cdef size_t i, j
    cdef float i_x, i_y, i_z, j_x, j_y, j_z

    for i in range(n_atoms):
        i_x = X_ptr[3*i + 0]
        i_y = X_ptr[3*i + 1]
        i_z = X_ptr[3*i + 2]
        for j in range(i+1, n_atoms):
            j_x = X_ptr[3*j + 0]
            j_y = X_ptr[3*j + 1]
            j_z = X_ptr[3*j + 2]
            r_ij = sqrt(pow(i_x-j_x, 2) + pow(i_y-j_y, 2) + pow(i_z-j_z, 2))
            dist_matrix[i*n_atoms+j] = r_ij
            dist_matrix[j*n_atoms+i] = r_ij

    cdef float *features    = <float*> &res[0]


    # TOTALLY WRONG
    # for i in range(n_atoms): # NO NEED TO DO AN EXTRA LOP
    radial_symmetry(&dist_matrix[0], features, n_atoms, Z_ptr)
    # angular_symmetry(&dist_matrix[0], features, X_ptr, n_atoms, Z_ptr)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ANI1_multi(float[:, :, ::1] X, int[::1] Z, float[::1] res) nogil:

    cdef int n_mols = X.shape[0]
    cdef int n_atoms = Z.shape[0]

    cdef float *X_ptr = &X[0][0][0]
    cdef int *Z_ptr    = <int*> &Z[0]
    cdef float *features = <float*> &res[0]

    cdef int mm

    with nogil, parallel():

        for mm in prange(n_mols, schedule='static'):

            # printf("processing %d\n", mm)
            # X_ptr = <float*> &X[mm][0][0]
            # mol_features = <float*> &res[mm*n_atoms*TOTAL_FEATURE_SIZE]

            # some pointer arithmetic to generate the correct offsets, 3 is for (x,y,z)
            ANI1_single(X_ptr+mm*n_atoms*3, Z_ptr, features+mm*n_atoms*TOTAL_FEATURE_SIZE, n_atoms)



        # dist_matrix = vector[float](n_atoms*n_atoms)

        # for i in range(n_atoms):
        #     i_x = X_ptr[3*i + 0]
        #     i_y = X_ptr[3*i + 1]
        #     i_z = X_ptr[3*i + 2]
        #     for j in range(i+1, n_atoms):
        #         j_x = X_ptr[3*j + 0]
        #         j_y = X_ptr[3*j + 1]
        #         j_z = X_ptr[3*j + 2]
        #         r_ij = sqrt(pow(i_x-j_x, 2) + pow(i_y-j_y, 2) + pow(i_z-j_z, 2))
        #         dist_matrix[i*n_atoms+j] = r_ij
        #         dist_matrix[j*n_atoms+i] = r_ij


        # radial_symmetry(&dist_matrix[0], features, n_atoms, Z_ptr)
        # angular_symmetry(&dist_matrix[0], features, X_ptr, n_atoms, Z_ptr)



    # return features

cdef size_t[4][4][8][4] A_map
A_map[0][0][0][0]=0
A_map[0][0][0][1]=1
A_map[0][0][0][2]=2
A_map[0][0][0][3]=3
A_map[0][0][1][0]=4
A_map[0][0][1][1]=5
A_map[0][0][1][2]=6
A_map[0][0][1][3]=7
A_map[0][0][2][0]=8
A_map[0][0][2][1]=9
A_map[0][0][2][2]=10
A_map[0][0][2][3]=11
A_map[0][0][3][0]=12
A_map[0][0][3][1]=13
A_map[0][0][3][2]=14
A_map[0][0][3][3]=15
A_map[0][0][4][0]=16
A_map[0][0][4][1]=17
A_map[0][0][4][2]=18
A_map[0][0][4][3]=19
A_map[0][0][5][0]=20
A_map[0][0][5][1]=21
A_map[0][0][5][2]=22
A_map[0][0][5][3]=23
A_map[0][0][6][0]=24
A_map[0][0][6][1]=25
A_map[0][0][6][2]=26
A_map[0][0][6][3]=27
A_map[0][0][7][0]=28
A_map[0][0][7][1]=29
A_map[0][0][7][2]=30
A_map[0][0][7][3]=31
A_map[0][1][0][0]=32
A_map[1][0][0][0]=32
A_map[0][1][0][1]=33
A_map[1][0][0][1]=33
A_map[0][1][0][2]=34
A_map[1][0][0][2]=34
A_map[0][1][0][3]=35
A_map[1][0][0][3]=35
A_map[0][1][1][0]=36
A_map[1][0][1][0]=36
A_map[0][1][1][1]=37
A_map[1][0][1][1]=37
A_map[0][1][1][2]=38
A_map[1][0][1][2]=38
A_map[0][1][1][3]=39
A_map[1][0][1][3]=39
A_map[0][1][2][0]=40
A_map[1][0][2][0]=40
A_map[0][1][2][1]=41
A_map[1][0][2][1]=41
A_map[0][1][2][2]=42
A_map[1][0][2][2]=42
A_map[0][1][2][3]=43
A_map[1][0][2][3]=43
A_map[0][1][3][0]=44
A_map[1][0][3][0]=44
A_map[0][1][3][1]=45
A_map[1][0][3][1]=45
A_map[0][1][3][2]=46
A_map[1][0][3][2]=46
A_map[0][1][3][3]=47
A_map[1][0][3][3]=47
A_map[0][1][4][0]=48
A_map[1][0][4][0]=48
A_map[0][1][4][1]=49
A_map[1][0][4][1]=49
A_map[0][1][4][2]=50
A_map[1][0][4][2]=50
A_map[0][1][4][3]=51
A_map[1][0][4][3]=51
A_map[0][1][5][0]=52
A_map[1][0][5][0]=52
A_map[0][1][5][1]=53
A_map[1][0][5][1]=53
A_map[0][1][5][2]=54
A_map[1][0][5][2]=54
A_map[0][1][5][3]=55
A_map[1][0][5][3]=55
A_map[0][1][6][0]=56
A_map[1][0][6][0]=56
A_map[0][1][6][1]=57
A_map[1][0][6][1]=57
A_map[0][1][6][2]=58
A_map[1][0][6][2]=58
A_map[0][1][6][3]=59
A_map[1][0][6][3]=59
A_map[0][1][7][0]=60
A_map[1][0][7][0]=60
A_map[0][1][7][1]=61
A_map[1][0][7][1]=61
A_map[0][1][7][2]=62
A_map[1][0][7][2]=62
A_map[0][1][7][3]=63
A_map[1][0][7][3]=63
A_map[0][2][0][0]=64
A_map[2][0][0][0]=64
A_map[0][2][0][1]=65
A_map[2][0][0][1]=65
A_map[0][2][0][2]=66
A_map[2][0][0][2]=66
A_map[0][2][0][3]=67
A_map[2][0][0][3]=67
A_map[0][2][1][0]=68
A_map[2][0][1][0]=68
A_map[0][2][1][1]=69
A_map[2][0][1][1]=69
A_map[0][2][1][2]=70
A_map[2][0][1][2]=70
A_map[0][2][1][3]=71
A_map[2][0][1][3]=71
A_map[0][2][2][0]=72
A_map[2][0][2][0]=72
A_map[0][2][2][1]=73
A_map[2][0][2][1]=73
A_map[0][2][2][2]=74
A_map[2][0][2][2]=74
A_map[0][2][2][3]=75
A_map[2][0][2][3]=75
A_map[0][2][3][0]=76
A_map[2][0][3][0]=76
A_map[0][2][3][1]=77
A_map[2][0][3][1]=77
A_map[0][2][3][2]=78
A_map[2][0][3][2]=78
A_map[0][2][3][3]=79
A_map[2][0][3][3]=79
A_map[0][2][4][0]=80
A_map[2][0][4][0]=80
A_map[0][2][4][1]=81
A_map[2][0][4][1]=81
A_map[0][2][4][2]=82
A_map[2][0][4][2]=82
A_map[0][2][4][3]=83
A_map[2][0][4][3]=83
A_map[0][2][5][0]=84
A_map[2][0][5][0]=84
A_map[0][2][5][1]=85
A_map[2][0][5][1]=85
A_map[0][2][5][2]=86
A_map[2][0][5][2]=86
A_map[0][2][5][3]=87
A_map[2][0][5][3]=87
A_map[0][2][6][0]=88
A_map[2][0][6][0]=88
A_map[0][2][6][1]=89
A_map[2][0][6][1]=89
A_map[0][2][6][2]=90
A_map[2][0][6][2]=90
A_map[0][2][6][3]=91
A_map[2][0][6][3]=91
A_map[0][2][7][0]=92
A_map[2][0][7][0]=92
A_map[0][2][7][1]=93
A_map[2][0][7][1]=93
A_map[0][2][7][2]=94
A_map[2][0][7][2]=94
A_map[0][2][7][3]=95
A_map[2][0][7][3]=95
A_map[0][3][0][0]=96
A_map[3][0][0][0]=96
A_map[0][3][0][1]=97
A_map[3][0][0][1]=97
A_map[0][3][0][2]=98
A_map[3][0][0][2]=98
A_map[0][3][0][3]=99
A_map[3][0][0][3]=99
A_map[0][3][1][0]=100
A_map[3][0][1][0]=100
A_map[0][3][1][1]=101
A_map[3][0][1][1]=101
A_map[0][3][1][2]=102
A_map[3][0][1][2]=102
A_map[0][3][1][3]=103
A_map[3][0][1][3]=103
A_map[0][3][2][0]=104
A_map[3][0][2][0]=104
A_map[0][3][2][1]=105
A_map[3][0][2][1]=105
A_map[0][3][2][2]=106
A_map[3][0][2][2]=106
A_map[0][3][2][3]=107
A_map[3][0][2][3]=107
A_map[0][3][3][0]=108
A_map[3][0][3][0]=108
A_map[0][3][3][1]=109
A_map[3][0][3][1]=109
A_map[0][3][3][2]=110
A_map[3][0][3][2]=110
A_map[0][3][3][3]=111
A_map[3][0][3][3]=111
A_map[0][3][4][0]=112
A_map[3][0][4][0]=112
A_map[0][3][4][1]=113
A_map[3][0][4][1]=113
A_map[0][3][4][2]=114
A_map[3][0][4][2]=114
A_map[0][3][4][3]=115
A_map[3][0][4][3]=115
A_map[0][3][5][0]=116
A_map[3][0][5][0]=116
A_map[0][3][5][1]=117
A_map[3][0][5][1]=117
A_map[0][3][5][2]=118
A_map[3][0][5][2]=118
A_map[0][3][5][3]=119
A_map[3][0][5][3]=119
A_map[0][3][6][0]=120
A_map[3][0][6][0]=120
A_map[0][3][6][1]=121
A_map[3][0][6][1]=121
A_map[0][3][6][2]=122
A_map[3][0][6][2]=122
A_map[0][3][6][3]=123
A_map[3][0][6][3]=123
A_map[0][3][7][0]=124
A_map[3][0][7][0]=124
A_map[0][3][7][1]=125
A_map[3][0][7][1]=125
A_map[0][3][7][2]=126
A_map[3][0][7][2]=126
A_map[0][3][7][3]=127
A_map[3][0][7][3]=127
A_map[1][1][0][0]=128
A_map[1][1][0][1]=129
A_map[1][1][0][2]=130
A_map[1][1][0][3]=131
A_map[1][1][1][0]=132
A_map[1][1][1][1]=133
A_map[1][1][1][2]=134
A_map[1][1][1][3]=135
A_map[1][1][2][0]=136
A_map[1][1][2][1]=137
A_map[1][1][2][2]=138
A_map[1][1][2][3]=139
A_map[1][1][3][0]=140
A_map[1][1][3][1]=141
A_map[1][1][3][2]=142
A_map[1][1][3][3]=143
A_map[1][1][4][0]=144
A_map[1][1][4][1]=145
A_map[1][1][4][2]=146
A_map[1][1][4][3]=147
A_map[1][1][5][0]=148
A_map[1][1][5][1]=149
A_map[1][1][5][2]=150
A_map[1][1][5][3]=151
A_map[1][1][6][0]=152
A_map[1][1][6][1]=153
A_map[1][1][6][2]=154
A_map[1][1][6][3]=155
A_map[1][1][7][0]=156
A_map[1][1][7][1]=157
A_map[1][1][7][2]=158
A_map[1][1][7][3]=159
A_map[1][2][0][0]=160
A_map[2][1][0][0]=160
A_map[1][2][0][1]=161
A_map[2][1][0][1]=161
A_map[1][2][0][2]=162
A_map[2][1][0][2]=162
A_map[1][2][0][3]=163
A_map[2][1][0][3]=163
A_map[1][2][1][0]=164
A_map[2][1][1][0]=164
A_map[1][2][1][1]=165
A_map[2][1][1][1]=165
A_map[1][2][1][2]=166
A_map[2][1][1][2]=166
A_map[1][2][1][3]=167
A_map[2][1][1][3]=167
A_map[1][2][2][0]=168
A_map[2][1][2][0]=168
A_map[1][2][2][1]=169
A_map[2][1][2][1]=169
A_map[1][2][2][2]=170
A_map[2][1][2][2]=170
A_map[1][2][2][3]=171
A_map[2][1][2][3]=171
A_map[1][2][3][0]=172
A_map[2][1][3][0]=172
A_map[1][2][3][1]=173
A_map[2][1][3][1]=173
A_map[1][2][3][2]=174
A_map[2][1][3][2]=174
A_map[1][2][3][3]=175
A_map[2][1][3][3]=175
A_map[1][2][4][0]=176
A_map[2][1][4][0]=176
A_map[1][2][4][1]=177
A_map[2][1][4][1]=177
A_map[1][2][4][2]=178
A_map[2][1][4][2]=178
A_map[1][2][4][3]=179
A_map[2][1][4][3]=179
A_map[1][2][5][0]=180
A_map[2][1][5][0]=180
A_map[1][2][5][1]=181
A_map[2][1][5][1]=181
A_map[1][2][5][2]=182
A_map[2][1][5][2]=182
A_map[1][2][5][3]=183
A_map[2][1][5][3]=183
A_map[1][2][6][0]=184
A_map[2][1][6][0]=184
A_map[1][2][6][1]=185
A_map[2][1][6][1]=185
A_map[1][2][6][2]=186
A_map[2][1][6][2]=186
A_map[1][2][6][3]=187
A_map[2][1][6][3]=187
A_map[1][2][7][0]=188
A_map[2][1][7][0]=188
A_map[1][2][7][1]=189
A_map[2][1][7][1]=189
A_map[1][2][7][2]=190
A_map[2][1][7][2]=190
A_map[1][2][7][3]=191
A_map[2][1][7][3]=191
A_map[1][3][0][0]=192
A_map[3][1][0][0]=192
A_map[1][3][0][1]=193
A_map[3][1][0][1]=193
A_map[1][3][0][2]=194
A_map[3][1][0][2]=194
A_map[1][3][0][3]=195
A_map[3][1][0][3]=195
A_map[1][3][1][0]=196
A_map[3][1][1][0]=196
A_map[1][3][1][1]=197
A_map[3][1][1][1]=197
A_map[1][3][1][2]=198
A_map[3][1][1][2]=198
A_map[1][3][1][3]=199
A_map[3][1][1][3]=199
A_map[1][3][2][0]=200
A_map[3][1][2][0]=200
A_map[1][3][2][1]=201
A_map[3][1][2][1]=201
A_map[1][3][2][2]=202
A_map[3][1][2][2]=202
A_map[1][3][2][3]=203
A_map[3][1][2][3]=203
A_map[1][3][3][0]=204
A_map[3][1][3][0]=204
A_map[1][3][3][1]=205
A_map[3][1][3][1]=205
A_map[1][3][3][2]=206
A_map[3][1][3][2]=206
A_map[1][3][3][3]=207
A_map[3][1][3][3]=207
A_map[1][3][4][0]=208
A_map[3][1][4][0]=208
A_map[1][3][4][1]=209
A_map[3][1][4][1]=209
A_map[1][3][4][2]=210
A_map[3][1][4][2]=210
A_map[1][3][4][3]=211
A_map[3][1][4][3]=211
A_map[1][3][5][0]=212
A_map[3][1][5][0]=212
A_map[1][3][5][1]=213
A_map[3][1][5][1]=213
A_map[1][3][5][2]=214
A_map[3][1][5][2]=214
A_map[1][3][5][3]=215
A_map[3][1][5][3]=215
A_map[1][3][6][0]=216
A_map[3][1][6][0]=216
A_map[1][3][6][1]=217
A_map[3][1][6][1]=217
A_map[1][3][6][2]=218
A_map[3][1][6][2]=218
A_map[1][3][6][3]=219
A_map[3][1][6][3]=219
A_map[1][3][7][0]=220
A_map[3][1][7][0]=220
A_map[1][3][7][1]=221
A_map[3][1][7][1]=221
A_map[1][3][7][2]=222
A_map[3][1][7][2]=222
A_map[1][3][7][3]=223
A_map[3][1][7][3]=223
A_map[2][2][0][0]=224
A_map[2][2][0][1]=225
A_map[2][2][0][2]=226
A_map[2][2][0][3]=227
A_map[2][2][1][0]=228
A_map[2][2][1][1]=229
A_map[2][2][1][2]=230
A_map[2][2][1][3]=231
A_map[2][2][2][0]=232
A_map[2][2][2][1]=233
A_map[2][2][2][2]=234
A_map[2][2][2][3]=235
A_map[2][2][3][0]=236
A_map[2][2][3][1]=237
A_map[2][2][3][2]=238
A_map[2][2][3][3]=239
A_map[2][2][4][0]=240
A_map[2][2][4][1]=241
A_map[2][2][4][2]=242
A_map[2][2][4][3]=243
A_map[2][2][5][0]=244
A_map[2][2][5][1]=245
A_map[2][2][5][2]=246
A_map[2][2][5][3]=247
A_map[2][2][6][0]=248
A_map[2][2][6][1]=249
A_map[2][2][6][2]=250
A_map[2][2][6][3]=251
A_map[2][2][7][0]=252
A_map[2][2][7][1]=253
A_map[2][2][7][2]=254
A_map[2][2][7][3]=255
A_map[2][3][0][0]=256
A_map[3][2][0][0]=256
A_map[2][3][0][1]=257
A_map[3][2][0][1]=257
A_map[2][3][0][2]=258
A_map[3][2][0][2]=258
A_map[2][3][0][3]=259
A_map[3][2][0][3]=259
A_map[2][3][1][0]=260
A_map[3][2][1][0]=260
A_map[2][3][1][1]=261
A_map[3][2][1][1]=261
A_map[2][3][1][2]=262
A_map[3][2][1][2]=262
A_map[2][3][1][3]=263
A_map[3][2][1][3]=263
A_map[2][3][2][0]=264
A_map[3][2][2][0]=264
A_map[2][3][2][1]=265
A_map[3][2][2][1]=265
A_map[2][3][2][2]=266
A_map[3][2][2][2]=266
A_map[2][3][2][3]=267
A_map[3][2][2][3]=267
A_map[2][3][3][0]=268
A_map[3][2][3][0]=268
A_map[2][3][3][1]=269
A_map[3][2][3][1]=269
A_map[2][3][3][2]=270
A_map[3][2][3][2]=270
A_map[2][3][3][3]=271
A_map[3][2][3][3]=271
A_map[2][3][4][0]=272
A_map[3][2][4][0]=272
A_map[2][3][4][1]=273
A_map[3][2][4][1]=273
A_map[2][3][4][2]=274
A_map[3][2][4][2]=274
A_map[2][3][4][3]=275
A_map[3][2][4][3]=275
A_map[2][3][5][0]=276
A_map[3][2][5][0]=276
A_map[2][3][5][1]=277
A_map[3][2][5][1]=277
A_map[2][3][5][2]=278
A_map[3][2][5][2]=278
A_map[2][3][5][3]=279
A_map[3][2][5][3]=279
A_map[2][3][6][0]=280
A_map[3][2][6][0]=280
A_map[2][3][6][1]=281
A_map[3][2][6][1]=281
A_map[2][3][6][2]=282
A_map[3][2][6][2]=282
A_map[2][3][6][3]=283
A_map[3][2][6][3]=283
A_map[2][3][7][0]=284
A_map[3][2][7][0]=284
A_map[2][3][7][1]=285
A_map[3][2][7][1]=285
A_map[2][3][7][2]=286
A_map[3][2][7][2]=286
A_map[2][3][7][3]=287
A_map[3][2][7][3]=287
A_map[3][3][0][0]=288
A_map[3][3][0][1]=289
A_map[3][3][0][2]=290
A_map[3][3][0][3]=291
A_map[3][3][1][0]=292
A_map[3][3][1][1]=293
A_map[3][3][1][2]=294
A_map[3][3][1][3]=295
A_map[3][3][2][0]=296
A_map[3][3][2][1]=297
A_map[3][3][2][2]=298
A_map[3][3][2][3]=299
A_map[3][3][3][0]=300
A_map[3][3][3][1]=301
A_map[3][3][3][2]=302
A_map[3][3][3][3]=303
A_map[3][3][4][0]=304
A_map[3][3][4][1]=305
A_map[3][3][4][2]=306
A_map[3][3][4][3]=307
A_map[3][3][5][0]=308
A_map[3][3][5][1]=309
A_map[3][3][5][2]=310
A_map[3][3][5][3]=311
A_map[3][3][6][0]=312
A_map[3][3][6][1]=313
A_map[3][3][6][2]=314
A_map[3][3][6][3]=315
A_map[3][3][7][0]=316
A_map[3][3][7][1]=317
A_map[3][3][7][2]=318
A_map[3][3][7][3]=319