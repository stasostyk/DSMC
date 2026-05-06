#ifndef DSMC_COLLIDER_H
#define DSMC_COLLIDER_H

#include "config.h"
#include "particle.h"

int collide_particles(
    Simulation *sim,
    Config *conf
);

#endif //DSMC_COLLIDER_H
