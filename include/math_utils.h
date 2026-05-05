#ifndef DSMC_MATH_UTILS_H
#define DSMC_MATH_UTILS_H

double randu();
double randn(double mean, double stddev);
double randp(double p);
void random_isotropic_vector (double *R);
void random_unit_vector(double *nx, double *ny, double *nz);
double rayleigh(double sigma);

void diffuse_scattering_y(
    double *vx, double *vy, double *vz, 
    double m, double T, double Ny,
    double KB
);

void diffuse_scattering(
    double *vx, double *vy, double *vz, 
    double m, double T, 
    double nx, double ny, double nz,
    double KB
);

#endif //DSMC_MATH_UTILS_H
