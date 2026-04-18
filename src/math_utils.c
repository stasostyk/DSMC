#include <stdlib.h>
#include <math.h>
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

double rayleigh(double sigma) {
    double u = randu();
    return sigma * sqrt(-2.0 * log(u));
}

void diffuse_scattering_y(double *vx, double *vy, double *vz, double m, double T, double Ny) {
    double RT = sqrt(KB * T / m);

    *vx = randn(0.0, RT);
    *vz = randn(0.0, RT);
    *vy = Ny * rayleigh(RT);
}
