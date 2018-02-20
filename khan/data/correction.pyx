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

cdef enum: n_atom_types = 4

cdef int triplet_indexes[n_atom_types][n_atom_types][n_atom_types]
triplet_index = 0
for i in range(n_atom_types):
    for j in range(i, n_atom_types):
        for k in range(j, n_atom_types):
            # skip some possible triplets not expected to be well-sampled in GDB-8, and thus too easy to overfit: HHH, NNN, NNO, NOO, OOO
            # if     (i==0 and j==0 and k==0) or \
            if     (i==2 and j==2 and k==2) or \
                   (i==2 and j==2 and k==3) or \
                   (i==2 and j==3 and k==3) or \
                   (i==3 and j==3 and k==3):
                triplet_indexes[i][j][k] = -1
                triplet_indexes[j][i][k] = -1
                triplet_indexes[k][i][j] = -1
                triplet_indexes[i][k][j] = -1
                triplet_indexes[j][k][i] = -1
                triplet_indexes[k][j][i] = -1
                continue
            # included triplets: get all combinations
            triplet_indexes[i][j][k] = triplet_index
            triplet_indexes[j][i][k] = triplet_index
            triplet_indexes[k][i][j] = triplet_index
            triplet_indexes[i][k][j] = triplet_index
            triplet_indexes[j][k][i] = triplet_index
            triplet_indexes[k][j][i] = triplet_index
            triplet_index += 1

cdef int n_triplets = triplet_index


cdef int pair_indexes[n_atom_types][n_atom_types]
pair_index = 0
for i in range(n_atom_types):
    for j in range(i, n_atom_types):
        # print("setting", i, j, "to", pair_index)
        pair_indexes[i][j] = pair_index
        pair_indexes[j][i] = pair_index
        pair_index += 1
cdef public int n_pairs = pair_index

cdef unsigned int n_pair_basis_functions = 4
cdef unsigned int n_triplet_basis_functions = 10

cdef double basis_cache[96][96]

cdef double solution[204]
# solution = [-320.24294, -23783.96561, -34309.19603, -47176.25721, -25.81252, -89.61229, -63.27550, -115.42429, -90.87959, 71.56868, 110.38722, 193.51917, 143.18840, 92.01213, 408.20392, 47.09845, 823.54091, 1373.49387, 244.23223, 442.75469, 507.63742, 1133.44876, 1715.69001, 765.46860, 70.79753, -1547.46590, -3627.70925, -4740.51776, -4357.61055, -6048.34059, -5983.67816, -8952.13153, -9660.84576, -7494.70902, -160.09947, 2229.69776, 3581.04011, 4140.91967, 6382.89336, 7785.27419, 7211.28164, 10925.46713, 10993.72807, 26828.75920, 563.54570, 53.10032, -1140.53013, 3179.98255, 1117.94858, -117.98047, -2800.98121, -4983.19477, -1471.21864, 630.82935, -3790.63616, -9709.52267, -5725.26916, -8634.33881, -9763.42988, -996.15768, -298.79685, 5814.67636, -4537.89634, -3569.14336, -2081.99009, 218.31602, 3343.61837, 5387.87141, 363.38712, 8040.57845, 22090.17724, 3322.87635, 12311.45938, 20839.58824, -530.44441, -275.21096, 2395.10253, -4608.65492, -3516.49013, -1926.24055, 498.43089, 2990.11270, 2295.01729, 2699.88102, 8095.50013, 21565.82659, 3572.32169, 15067.15184, 22052.21360, -539.02639, -317.05074, 2275.81921, 984.71288, 4318.94134, 6826.41033, 10786.19343, 16054.75069, -40017.77798, 762.01487, 8645.06818, 22181.48110, 6101.85033, 14381.98109, 23141.76074, -48.01947, -170.41660, -17782.47753, 10080.38146, 15405.55441, 12119.91605, 19201.39995, 16484.01224, -76515.92794, -1976.41306, -9556.34620, -41736.53368, 33132.58014, -7528.39063, -53437.85297, -30.34487, -24.89329, -17419.64925, -5996.01694, -4261.98667, -6031.14857, -3573.92409, -10228.53159, 105880.89820, 4043.44980, -10820.05951, -43040.87656, 29392.60335, -5227.89859, -55942.85251, -755.32341, 227.85151, -3747.01491, -5776.78756, -4413.84531, -6463.48778, -4294.08010, -9346.21102, 14242.25538, -2849.28854, -12438.06145, -42517.45522, 26743.49148, -12038.44946, -58879.18848, 5621.66975, -495.18368, 45269.77008, 51116.87480, 8324.53362, -425.97027, -79276.78974, -82523.27085, 726603.32252, 20736.77342, 26625.49465, 82747.31633, -397204.14896, -144592.87921, 293183.89638, -18310.37448, 5121.24564, 14395.73806, -425994.35928, -157954.71371, -16331.26567, 519520.12769, 544435.49566, -20070392.46386, -1047968.73605, -550602.59708, -152357.12557, 5433484.53302, 2276332.59672, -2198765.77601, 34272.12515, -5308.42881, -195509.50942, 1507756.86563, 545660.79633, 54222.29619, -1725394.84349, -1586047.53062, 203681910.73278, 6173186.16618, 2639295.66164, 629880.45063, -41933738.17881, -14164723.97647, 10596340.03113]

# solution = [-313.24147, -23782.89248, -34316.34818, -47177.17761, -10.81821, -98.08305, -62.76133, -107.08538, -72.24847, 91.63982, 108.31009, 196.67520, 148.96673, 133.31214, 264.56872, -105.49744, 705.39878, 1198.39238, 135.22481, 445.90409, 564.36358, 1454.30104, 1719.34548, 780.81501, 201.70271, -1143.33038, -3349.54065, -4350.81417, -4162.78322, -6188.14133, -6178.28597, -9965.74016, -9607.85163, -4058.97988, -234.57860, 1967.77764, 3417.31189, 3898.68185, 6270.86491, 7993.38957, 7395.53670, 11585.52450, 10858.21284, 4932.29612, 470.00845, 589.41505, -826.62060, 3577.77684, 1356.51963, 33.48854, -5673.35919, -6653.95316, -4361.44769, 181.51813, -4248.74640, -9464.84460, -4464.08588, -7513.99437, -7603.59011, -359.11883, -1872.58002, 4307.34933, -5058.48475, -4071.35005, -2462.85175, 3385.56339, 4515.97767, 2594.88268, 710.45522, 8986.45408, 21166.12033, 1381.57834, 10153.68914, 14983.06754, -302.30062, -888.79671, 2663.23237, -5056.42554, -4008.86116, -2311.30315, 3833.40489, 4219.63059, 1240.74258, 3164.17338, 9555.73971, 20702.60302, 37.29574, 11420.17987, 14526.43273, -334.89065, -854.10048, 2549.71000, 948.04105, 3610.78471, 6651.57498, 17537.38263, 19579.97856, 7749.72239, 1621.22445, 9728.70683, 21416.54278, 3585.68966, 11547.53978, 17066.28725, -768.12088, 1887.56435, -19161.42060, 10318.93873, 16597.67317, 13493.97475, 19149.76720, 19759.87370, -6991.45891, -310.57599, -14332.53102, -38752.21265, 37706.62807, 302.64262, -31687.75220, -676.80037, 1815.23199, -18348.52805, -6385.94181, -2806.72304, -5362.19894, -10119.09793, -11464.52881, -11661.00733, 3197.36970, -14428.96904, -40792.59329, 31913.75855, 976.39890, -41663.68539, -1313.21046, 447.05344, -10012.85513, -6403.24221, -2953.33153, -5805.64964, -11202.28500, -10436.11491, -7333.78582, -4351.18753, -17458.09925, -40534.35728, 32607.47753, -2781.87604, -41279.90582, 7034.57986, -1128.53089, 86068.44982, 57297.70093, 4462.22861, -2587.50278, -90130.56738, -100560.59814, 149133.23208, 21565.64047, 50943.76050, 79761.75282, -414477.28484, -191366.60568, 250757.29053, -19816.30742, 460.61008, -304453.81871, -470505.97578, -144480.82869, -19372.02357, 637472.58531, 596518.98078, -2309156.20008, -1117259.84952, -659644.40500, -217857.17154, 6649079.77076, 3331945.64424, -2416531.81593, 36377.14904, -467.00567, 712681.79649, 1661067.33368, 478300.22026, 61240.97136, -2143875.33940, -1624051.18676, 15061127.73044, 6398104.03051, 2997117.28863, 923283.07121, -57504370.26855, -24108234.36749, 12885662.30698]

solution = [-313.32885, -23782.26501, -34316.47118, -47177.35326, -3.29421, -99.99155, -64.77073, -106.88987, -72.58318, 93.50313, 108.64322, 198.15856, 148.81629, 133.70529, 208.55242, -89.99276, 720.88717, 1201.11668, 125.69799, 428.75244, 554.31064, 1436.30994, 1720.41562, 785.83907, 253.62568, -1177.35417, -3383.32690, -4357.31122, -4130.22044, -6141.53918, -6142.91082, -9913.20075, -9609.28678, -4098.53106, -265.96686, 1986.24019, 3439.88227, 3903.08736, 6233.12496, 7950.36524, 7362.82559, 11544.86207, 10863.00414, 4992.59570, 3123.29837, -113.08059, 153.88388, -1096.42165, 3820.84073, 1542.81332, 198.07900, -5403.28964, -6542.26408, -4353.85109, 59.51256, -4263.03098, -9389.95060, -4515.63852, -7526.14913, -7511.92790, -11458.89952, 1374.32249, -508.16380, 5001.19390, -5541.72027, -4408.40476, -2763.77548, 2965.26624, 4338.37486, 2545.05575, 1116.71024, 9022.25565, 20961.98653, 1489.41807, 10167.20183, 14746.17598, -12361.00131, 556.47848, -223.41655, 3015.99312, -5512.20480, -4342.73534, -2606.69263, 3412.02948, 4029.96456, 1134.01926, 3686.98434, 9618.10573, 20544.90926, 181.64055, 11490.22299, 14332.14195, -10669.86756, 507.00380, -217.45764, 2863.80339, 481.22676, 3258.29574, 6298.38759, 16970.59072, 19309.90861, 7753.77362, 2002.79453, 9747.46163, 21236.50901, 3683.62646, 11556.77089, 16823.93187, 40130.81444, -2811.50086, 95.74582, -19830.05732, 11278.45255, 17104.54731, 13952.29681, 19680.09889, 20057.58665, -6867.91217, -1989.30845, -14666.89934, -38474.30650, 37256.11290, 36.85216, -30815.72861, 34874.11125, -2676.30155, 93.42117, -18900.90184, -5461.34380, -2188.83519, -4773.37641, -9249.57335, -11029.95556, -11712.40062, 2017.32563, -14526.05527, -40343.99490, 31668.28870, 939.31104, -41071.17052, 37859.44917, -2545.04151, -492.29556, -10404.75869, -5542.00001, -2340.66852, -5218.04644, -10335.10644, -9981.38400, -7217.01845, -5925.60050, -17657.44268, -40194.50425, 32236.40539, -2957.62031, -40757.50719, -117196.36385, 9394.01225, 880.47761, 86072.64733, 56299.10615, 4097.28770, -2885.16644, -90952.43926, -101277.06203, 151000.98044, 27467.12110, 53036.73731, 79966.11301, -412186.91852, -189705.27662, 248361.44814, 95385.74298, -20535.36092, 2292.64853, -304510.40784, -483407.50223, -150910.22958, -25086.80259, 633615.98331, 595794.05106, -2344126.16712, -1135785.17151, -680512.00178, -229227.64577, 6608426.74874, 3304954.91299, -2408652.23968, -415965.10749, 37566.23670, -3386.82118, 723741.03972, 1710628.50151, 497543.49564, 76623.85317, -2130311.23911, -1620651.99393, 15265189.63165, 6438797.29284, 3082627.66143, 965879.46431, -57108083.90945, -23899390.80908, 12850144.95017]

@cython.boundscheck(False)
@cython.wraparound(False)
def jamesTripletCorrection_C(
    float[:, ::1] X,
    int[::1] Z):

    cdef float *X_ptr  = <float*> &X[0][0]
    cdef int *Z_ptr    = <int*> &Z[0]

    cdef int n_atoms = Z.shape[0]
    cdef int n_geometries = 1
    # loop indexes
    cdef int geom_index
    cdef int i, j, k
    # element indexes (from 0 to n_atom_types-1)
    cdef int element_i, element_j, element_k
    # geometric terms
    cdef double i_x, i_y, i_z
    cdef double j_x, j_y, j_z
    cdef double k_x, k_y, k_z
    cdef double r2_ij, r2_ik, r2_jk
    cdef double ij, ik, jk
    # index of a given pair of elements within the basis vector
    cdef int pair_index
    cdef int triplet_index
    cdef int point_index
    cdef int pow_ij, pow_ik, pow_jk, pow_index
    # result
    cdef double energy
    # the main loop
    # calculate gaussian basis functions
    #for geom_index in prange(n_geometries, nogil=True): # parallel for-loop. doesn't help speed in practice: process overhead? memory locality issues?
    # for geom_index in range(n_geometries):
        # to use weights, multipy each value in A and B with 1/(expected std dev of corresponding value in B). Keep sum of weights to post-correct RMSE, which will be scaled too. 

        # if EE_ptr[geom_index] > min_E + 300.0: # use energy cutoff
            # continue # very high points get no energy or params, stay at 0
    # print(pair_indexes)

    # print(n_atom_types, n_pair_basis_functions, n_pairs, n_triplets)

    energy = 0.0
    for i in range(n_atoms):
        element_i = Z_ptr[i]
        # count how much of each element is present
        energy += solution[ element_i ]
        # start adding up pairs
        i_x = X_ptr[3*i + 0]
        i_y = X_ptr[3*i + 1]
        i_z = X_ptr[3*i + 2]
        for j in range(i+1, n_atoms):
            element_j = Z_ptr[j]
            j_x = X_ptr[3*j + 0]
            j_y = X_ptr[3*j + 1]
            j_z = X_ptr[3*j + 2]
            r2_ij = pow(i_x-j_x, 2) + pow(i_y-j_y, 2) + pow(i_z-j_z, 2)
            ij = exp(-0.5*r2_ij)

            # print("EI", element_i, element_j)
            pair_index = pair_indexes[element_i][element_j]

            energy += solution[(n_atom_types + 0*n_pairs + pair_index)] * ij
            energy += solution[(n_atom_types + 1*n_pairs + pair_index)] * pow(ij, 2)
            energy += solution[(n_atom_types + 2*n_pairs + pair_index)] * pow(ij, 3)
            energy += solution[(n_atom_types + 3*n_pairs + pair_index)] * pow(ij, 4)

            basis_cache[i][j] = ij

    # used cached distances and calculate triples
    for i in range(n_atoms):
        element_i = Z_ptr[i]
        for j in range(i+1, n_atoms):
            element_j = Z_ptr[j]
            for k in range(j+1, n_atoms):
                element_k = Z_ptr[k]
                triplet_index = triplet_indexes[element_i][element_j][element_k]
                if triplet_index == -1: # this means skip
                    continue

                # guaranteed that i < j && i < k && j < k
                ij = basis_cache[i][j]
                ik = basis_cache[i][k]
                jk = basis_cache[j][k]

                pow_index = 0

                # print(n_atom_types, n_pair_basis_functions, n_pairs, pow_index, n_triplets, triplet_index)
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 1) * pow(ik, 1) * pow(jk, 1)
                pow_index += 1
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 1) * pow(ik, 1) * pow(jk, 2)
                pow_index += 1
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 1) * pow(ik, 2) * pow(jk, 1)
                pow_index += 1
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 2) * pow(ik, 1) * pow(jk, 1)
                pow_index += 1
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 1) * pow(ik, 2) * pow(jk, 2)
                pow_index += 1
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 2) * pow(ik, 1) * pow(jk, 2)
                pow_index += 1
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 2) * pow(ik, 2) * pow(jk, 1)
                pow_index += 1
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 2) * pow(ik, 2) * pow(jk, 2)
                pow_index += 1
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 3) * pow(ik, 3) * pow(jk, 3)
                pow_index += 1
                energy += solution[n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index] * pow(ij, 4) * pow(ik, 4) * pow(jk, 4)



                # pow_index = 0
                # for pow_ij in range(1,5):
                #     for pow_ik in range(1,5):
                #         for pow_jk in range(1,5):
                #             energy += solution[(n_atom_types + n_pair_basis_functions*n_pairs + pow_index*n_triplets + triplet_index)] * pow(ij, pow_ij) * pow(ik, pow_ik) * pow(jk, pow_jk)
                #             pow_index += 1

    return energy

        # sum_squared_errors += pow(EE_ptr[geom_index] - energy, 2)
        # count += 1
        # if count % 100000 == 0 or count == n_points:
            # printf("%10u  %10f\n", count, pow(sum_squared_errors/count,0.5))