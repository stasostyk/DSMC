#ifndef DSMC_COLLIDER_H
#define DSMC_COLLIDER_H

#include "config.h"
#include "particle.h"

int collide_particles(Particle *P,
                      int *cellCount,
                      int *cellList,
                      double weight,
                      double cellVolume);

void collider_setup();

#endif //DSMC_COLLIDER_H
