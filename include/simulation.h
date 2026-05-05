#ifndef SIMULATION_H
#define SIMULATION_H

#include "particle.h"
#include "cell.h"
#include "config.h"

typedef struct {
    Particle P[MAX_PARTICLES];
    Cell samples[NX][NY][NZ];
    int cellCount[NX][NY][NZ];
    int cellList[NX][NY][NZ][MAX_PARTICLES_PER_CELL];

    // counters
    int sampleSteps;
    int NP;
    long long totalCollisions;

    // derived quantities
    double dx, dy, dz;
    double cellVolume;
    double weight;
    double NFree;
    double UFree;
    double UxFree, UyFree, UzFree;
} Simulation;

void setup(Simulation *sim);
void index_particles(Simulation *sim);
void initialize_particles(Simulation *sim);
void apply_boundary_conditions_free_stream(Simulation *sim);
void move_particles(Simulation *sim);
void accumulate_sampling(Simulation *sim);
void clearPointers(Simulation *sim);

#endif // SIMULATION_H