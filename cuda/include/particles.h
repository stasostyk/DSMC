#ifndef DSMC_PARTICLES_H
#define DSMC_PARTICLES_H

typedef struct {
    double *x, *y, *z;
    double *vx, *vy, *vz;
} Particles;

#endif // DSMC_PARTICLES_H