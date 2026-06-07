#ifndef DSMC_PARTICLES_H
#define DSMC_PARTICLES_H

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} Particles;

#endif // DSMC_PARTICLES_H