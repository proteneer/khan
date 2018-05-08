#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#ifdef GOOGLE_CUDA
    #define CONSTANT_FLAGS __device__
#else
    #define CONSTANT_FLAGS
#endif

const float CHARGE_CONSTANT = 0.529176917;

/* from github repo */
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

CONSTANT_FLAGS const float R_Rs[NUM_R_Rs] = {
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
    5.0000000e-01,
    1.1500000e+00,
    1.8000000e+00,
    2.4500000e+00,
};

/* from ANI-1 paper */


// const int MAX_ATOM_TYPES = 4;

// const int NUM_R_Rs = 32;
// const int RADIAL_FEATURE_SIZE = MAX_ATOM_TYPES * NUM_R_Rs;

// const float R_eta = 16.0;
// const float R_Rc = 3.96;

// const float A_Rc = 3.1;
// const float A_eta = 6.0;
// const float A_zeta = 8.0;
// const int NUM_A_THETAS = 8;
// const int NUM_A_RS = 8;

// const int ANGULAR_FEATURE_SIZE = NUM_A_RS * NUM_A_THETAS * (MAX_ATOM_TYPES * (MAX_ATOM_TYPES+1) / 2);

// const int TOTAL_FEATURE_SIZE = RADIAL_FEATURE_SIZE + ANGULAR_FEATURE_SIZE;

// // portably transfer over to CPU code?
// CONSTANT_FLAGS const float R_Rs[NUM_R_Rs] = {
//     0.12,
//     0.24,
//     0.36,
//     0.48,
//     0.6,
//     0.72,
//     0.84,
//     0.96,
//     1.08,
//     1.2,
//     1.32,
//     1.44,
//     1.56,
//     1.68,
//     1.8,
//     1.92,
//     2.04,
//     2.16,
//     2.28,
//     2.4,
//     2.52,
//     2.64,
//     2.76,
//     2.88,
//     3,
//     3.12,
//     3.24,
//     3.36,
//     3.48,
//     3.6,
//     3.72,
//     3.84
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
//     0.34444444,
//     0.68888889,
//     1.03333333,
//     1.37777778,
//     1.72222222,
//     2.06666667,
//     2.41111111,
//     2.75555556
// };

#endif

