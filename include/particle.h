#include "config.h"

#ifndef DSMC_PARTICLE_H
#define DSMC_PARTICLE_H

typedef struct {
    double x, y;
    double vx, vy, vz;
} Particle;

#endif // DSMC_PARTICLE_H