#ifndef DSMC_MATH_UTILS_H
#define DSMC_MATH_UTILS_H

#include <curand_kernel.h>

double randu();
double randn(double mean, double stddev);
double randp(double p);
void random_isotropic_vector (double *R);
void random_unit_vector(double *nx, double *ny, double *nz);
double rayleigh(double sigma);

void diffuse_scattering_y(
    float *vx, float *vy, float *vz,
    double m, double T, double Ny,
    double KB
);

void diffuse_scattering(
    float *vx, float *vy, float *vz,
    double m, double T, 
    double nx, double ny, double nz,
    double KB
);


__device__
void diffuse_scattering_y_device(
    float *vx, float *vy, float *vz, double m, double T, double Ny, double KB,
    curandState *rngState
);

__device__
void diffuse_scattering_device(
    float *vx, float *vy, float *vz,
    double m, double T, 
    double nx, double ny, double nz,
    double KB,
    curandState *rngState
);

__device__
void random_isotropic_vector_device (float *R, curandState *rngState);

#endif //DSMC_MATH_UTILS_H
