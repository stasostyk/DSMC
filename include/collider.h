#ifndef DSMC_COLLIDER_H
#define DSMC_COLLIDER_H

#include "particle.h"

int collide_particles(Particle *P,
                      int cellCount[NX][NY],
                      int cellList[NX][NY][MAX_PARTICLES_PER_CELL],
                      double weight,
                      double cellVolume);

void collider_setup();

#endif //DSMC_COLLIDER_H
