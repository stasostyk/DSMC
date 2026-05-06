#ifndef SIMULATION_H
#define SIMULATION_H

#include "particle.h"
#include "cell.h"
#include "config.h"

// Only include CUDA headers when compiling with nvcc
#ifdef __CUDACC__
#include <curand_kernel.h>
typedef curandState RNGState;
#else
// CPU fallback type
typedef void* RNGState;
#endif

#define IDX_CELL(k, l, m) ((k)*NY*NZ + (l)*NZ + m)
#define IDX_LIST(k, l, m, q) (IDX_CELL(k, l, m) * MAX_PARTICLES_PER_CELL + q)

typedef struct {
    RNGState *rngStates; // for using randomness in GPU

    Particle *P;   // used by host (CPU)
    Particle *d_P; // used by device (GPU)

    Cell *samples;
    int *cellCount;
    int *cellList;

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

void setup(Simulation *sim, Config *conf);
void index_particles(Simulation *sim);
void initialize_particles(Simulation *sim, Config *conf);
void apply_boundary_conditions_free_stream(Simulation *sim, Config *conf);
void move_particles(Simulation *sim, Config *conf);
void accumulate_sampling(Simulation *sim);
void clearPointers(Simulation *sim);

#endif // SIMULATION_H