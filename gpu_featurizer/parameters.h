

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#ifdef GOOGLE_CUDA
    #define CONSTANT_FLAGS __device__
#else
    #define CONSTANT_FLAGS
#endif
/* from github repo */
// const int MAX_ATOM_TYPES = 4;

// const int NUM_R_Rs = 16;
// const int RADIAL_FEATURE_SIZE = MAX_ATOM_TYPES * NUM_R_Rs;

// const float R_eta = 16;
// const float R_Rc = 4.6;

// const float A_Rc = 3.1;
// const float A_eta = 6.0;
// const float A_zeta = 8.0;
// const int NUM_A_THETAS = 8;
// const int NUM_A_RS = 4;

// const int ANGULAR_FEATURE_SIZE = NUM_A_RS * NUM_A_THETAS * (MAX_ATOM_TYPES * (MAX_ATOM_TYPES+1) / 2);

// const int TOTAL_FEATURE_SIZE = RADIAL_FEATURE_SIZE + ANGULAR_FEATURE_SIZE;

// CONSTANT_FLAGS const float R_Rs[NUM_R_Rs] = {
//     5.0000000e-01,
//     7.5625000e-01,
//     1.0125000e+00,
//     1.2687500e+00,
//     1.5250000e+00,
//     1.7812500e+00,
//     2.0375000e+00,
//     2.2937500e+00,
//     2.5500000e+00,
//     2.8062500e+00,
//     3.0625000e+00,
//     3.3187500e+00,
//     3.5750000e+00,
//     3.8312500e+00,
//     4.0875000e+00,
//     4.3437500e+00
// };

// CONSTANT_FLAGS const float A_thetas[NUM_A_THETAS] = {
//     0.0000000e+00,
//     7.8539816e-01,
//     1.5707963e+00,
//     2.3561945e+00,
//     3.1415927e+00,
//     3.9269908e+00,
//     4.7123890e+00,
//     5.4977871e+00
// };

// CONSTANT_FLAGS const float A_Rs[NUM_A_RS] = {
//     5.0000000e-01,
//     1.1500000e+00,
//     1.8000000e+00,
//     2.4500000e+00,
// };

/* from ANI-1 paper */


const int MAX_ATOM_TYPES = 4;

const int NUM_R_Rs = 32;
const int RADIAL_FEATURE_SIZE = MAX_ATOM_TYPES * NUM_R_Rs;

const float R_eta = 16.0;
const float R_Rc = 4.6;

const float A_Rc = 3.1;
const float A_eta = 6.0;
const float A_zeta = 8.0;
const int NUM_A_THETAS = 8;
const int NUM_A_RS = 8;

const int ANGULAR_FEATURE_SIZE = NUM_A_RS * NUM_A_THETAS * (MAX_ATOM_TYPES * (MAX_ATOM_TYPES+1) / 2);

const int TOTAL_FEATURE_SIZE = RADIAL_FEATURE_SIZE + ANGULAR_FEATURE_SIZE;

// portably transfer over to CPU code?
CONSTANT_FLAGS const float R_Rs[NUM_R_Rs] = {
    0.13939394,
    0.27878788,
    0.41818182,
    0.55757576,
    0.6969697,
    0.83636364,
    0.97575758,
    1.11515152,
    1.25454545,
    1.39393939,
    1.53333333,
    1.67272727,
    1.81212121,
    1.95151515,
    2.09090909,
    2.23030303,
    2.36969697,
    2.50909091,
    2.64848485,
    2.78787879,
    2.92727273,
    3.06666667,
    3.20606061,
    3.34545455,
    3.48484848,
    3.62424242,
    3.76363636,
    3.9030303,
    4.04242424,
    4.18181818,
    4.32121212,
    4.46060606
};

CONSTANT_FLAGS const float A_thetas[NUM_A_THETAS] = {
    0.0000000e+00,
    7.8539816e-01,
    1.5707963e+00,
    2.3561945e+00,
    3.1415927e+00,
    3.9269908e+00,
    4.7123890e+00,
    5.4977871e+00
};

CONSTANT_FLAGS const float A_Rs[NUM_A_RS] = {
    0.34444444,
    0.68888889,
    1.03333333,
    1.37777778,
    1.72222222,
    2.06666667,
    2.41111111,
    2.75555556
};


#endif