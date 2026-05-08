#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include "../include/math_utils.h"
#include "../include/config.h"

double randu() {
    return (double)rand() / (double)RAND_MAX;
}

double randn(double mean, double stddev) {
    double u1 = randu();
    double u2 = randu();

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    return mean + stddev * r * cos(theta);
}

double randp(double lambda) {
    double u = randu();
    return -log(1.0 - u) / lambda;
}

void random_unit_vector(double *nx, double *ny, double *nz) {
    double mu = 2.0 * randu() - 1.0;
    double phi = 2.0 * M_PI * randu();
    double s = sqrt(1.0 - mu * mu);

    *nx = s * cos(phi);
    *ny = s * sin(phi);
    *nz = mu;
}

void random_isotropic_vector (double *R)
{
    double cosT = 1.0 - 2.0 * randu();
    double sinT = sqrt (1.0 - cosT * cosT);
    double E = 2.0 * M_PI * randu();
    R[0] = cosT;
    R[1] = sinT * cos(E);
    R[2] = sinT * sin(E);
}

__device__
void random_isotropic_vector_device (double *R, curandStatePhilox4_32_10_t *rngState)
{
    double cosT = 1.0 - 2.0 * curand_uniform(rngState);
    double sinT = sqrt (1.0 - cosT * cosT);
    double E = 2.0 * M_PI * curand_uniform(rngState);
    R[0] = cosT;
    R[1] = sinT * cos(E);
    R[2] = sinT * sin(E);
}

double rayleigh(double sigma) {
    double u = randu();
    return sigma * sqrt(-2.0 * log(u));
}

__device__
double rayleigh_device(double sigma, curandStatePhilox4_32_10_t *rngState) {
    double u = curand_uniform(rngState); // randu()
    return sigma * sqrt(-2.0 * log(u));
}

__device__
void diffuse_scattering_y_device(
    double *vx, double *vy, double *vz, double m, double T, double Ny, double KB,
    curandStatePhilox4_32_10_t *rngState
) {
    double RT = sqrt(KB * T / m);

    *vx = curand_normal(rngState) * RT; // randn(0.0, RT)
    *vz = curand_normal(rngState) * RT; // randn(0.0, RT)
    *vy = Ny * rayleigh_device(RT, rngState);
}

void diffuse_scattering_y(double *vx, double *vy, double *vz, double m, double T, double Ny, double KB) {
    double RT = sqrt(KB * T / m);

    *vx = randn(0.0, RT);
    *vz = randn(0.0, RT);
    *vy = Ny * rayleigh(RT);
}

void diffuse_scattering(
    double *vx, double *vy, double *vz, 
    double m, double T, 
    double nx, double ny, double nz,
    double KB
) {
    double RT = sqrt(KB * T / m);

    // normalize the norm
    double norm = sqrt(nx*nx + ny*ny + nz*nz);
    nx /= norm;
    ny /= norm;
    nz /= norm;

    // orthonormal basis
    double tx1, ty1, tz1;

    // pick a vector not parallel to n
    if (fabs(nx) < 0.9) {
        tx1 = 0.0;
        ty1 = -nz;
        tz1 = ny;
    } else {
        tx1 = -ny;
        ty1 = nx;
        tz1 = 0.0;
    }

     // normalize t1
    double t1norm = sqrt(tx1*tx1 + ty1*ty1 + tz1*tz1);
    tx1 /= t1norm;
    ty1 /= t1norm;
    tz1 /= t1norm;

    // t2 = n × t1
    double tx2 = ny*tz1 - nz*ty1;
    double ty2 = nz*tx1 - nx*tz1;
    double tz2 = nx*ty1 - ny*tx1;

    // sample velocities in local frame
    double v_t1 = randn(0.0, RT);
    double v_t2 = randn(0.0, RT);
    double v_n  = rayleigh(RT);   // always outward

    // transform back to global coordinates 
    *vx = v_t1 * tx1 + v_t2 * tx2 + v_n * nx;
    *vy = v_t1 * ty1 + v_t2 * ty2 + v_n * ny;
    *vz = v_t1 * tz1 + v_t2 * tz2 + v_n * nz;
}

__device__
void diffuse_scattering_device(
    double *vx, double *vy, double *vz, 
    double m, double T, 
    double nx, double ny, double nz,
    double KB,
    curandStatePhilox4_32_10_t *rngState
) {
    double RT = sqrt(KB * T / m);

    // normalize the norm
    double norm = sqrt(nx*nx + ny*ny + nz*nz);
    nx /= norm;
    ny /= norm;
    nz /= norm;

    // orthonormal basis
    double tx1, ty1, tz1;

    // pick a vector not parallel to n
    if (fabs(nx) < 0.9) {
        tx1 = 0.0;
        ty1 = -nz;
        tz1 = ny;
    } else {
        tx1 = -ny;
        ty1 = nx;
        tz1 = 0.0;
    }

     // normalize t1
    double t1norm = sqrt(tx1*tx1 + ty1*ty1 + tz1*tz1);
    tx1 /= t1norm;
    ty1 /= t1norm;
    tz1 /= t1norm;

    // t2 = n × t1
    double tx2 = ny*tz1 - nz*ty1;
    double ty2 = nz*tx1 - nx*tz1;
    double tz2 = nx*ty1 - ny*tx1;

    // sample velocities in local frame
    double v_t1 = curand_normal(rngState) * RT; // randn(0.0, RT)
    double v_t2 = curand_normal(rngState) * RT; // randn(0.0, RT)
    double v_n  = rayleigh_device(RT, rngState);   // always outward

    // transform back to global coordinates 
    *vx = v_t1 * tx1 + v_t2 * tx2 + v_n * nx;
    *vy = v_t1 * ty1 + v_t2 * ty2 + v_n * ny;
    *vz = v_t1 * tz1 + v_t2 * tz2 + v_n * nz;
}
