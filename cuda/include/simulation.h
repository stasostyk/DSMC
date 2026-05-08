#ifndef SIMULATION_H
#define SIMULATION_H

#include "particle.h"
#include "cell.h"
#include "config.h"

#include <curand_kernel.h>

#define IDX_CELL(k, l, m) ((k)*NY*NZ + (l)*NZ + m)
#define IDX_LIST(k, l, m, q) (IDX_CELL(k, l, m) * MAX_PARTICLES_PER_CELL + q)

typedef struct {
    // for using randomness in GPU
    // allocated in device memory (GPU)
    rngStateType *rngStates; 

    Config *conf;   // allocated in host (CPU)

    // Particles will be initialized by device kernels, and during simulation
    // will be dealt only by kernels (GPU). There is no need to allocate
    // the same particle data on CPU and keep moving data.
    // Particles in CPU are only moved to count and print global diagnostics data.
    // Similar logic applies to samples, cellCount, cellList.
    Particle *P;   // allocated in host memory (CPU)
    Particle *d_P; // allocated in device memory (GPU)
    Cell *samples;    // allocated in host memory (CPU)
    Cell *d_samples;  // allocated in device memory (GPU)
    int *d_cellCount; // allocated in device memory (GPU)
    int *d_cellList;  // allocated in device memory (GPU)

    // counters
    int sampleSteps;
    int NP;
    long long totalCollisions;
} Simulation;

void setup(Simulation *sim, Config *conf);
void index_particles(Simulation *sim);
void initialize_particles(Simulation *sim);
void apply_boundary_conditions_free_stream(Simulation *sim);
void move_particles(Simulation *sim);
void accumulate_sampling(Simulation *sim);
void clearPointers(Simulation *sim);

#endif // SIMULATION_H