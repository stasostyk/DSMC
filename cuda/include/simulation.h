#ifndef SIMULATION_H
#define SIMULATION_H

#include "particles.h"
#include "cell.h"
#include "config.h"
#include "mpi_helper.h"

#include <curand_kernel.h>

#define IDX_CELL(k, l, m) ((k)*NY*NZ + (l)*NZ + (m))
#define IDX_LIST(k, l, m, q) (IDX_CELL(k, l, m) * MAX_PARTICLES_PER_CELL + (q))

typedef struct {
    // for using randomness in GPU
    // allocated in device memory (GPU)
    curandState *rngStates; 

    Config *conf;   // allocated in host (CPU)

    // Particles will be initialized by device kernels, and during simulation
    // will be dealt only by kernels (GPU). There is no need to allocate
    // the same particle data on CPU and keep moving data.
    // Particles in CPU are only moved to count and print global diagnostics data.
    // Similar logic applies to samples, cellCount, cellList.
    Particles P;   // allocated in host memory (CPU)
    Particles d_P; // allocated in device memory (GPU)
    Particles d_new_P; // allocated in device memory(GPU), will be used as temporary storage
                       // when creating new particles
    Cell *samples;    // allocated in host memory (CPU)
    Cell *d_samples;  // allocated in device memory (GPU)
    int *d_cellCount; // allocated in device memory (GPU)
    int *d_cellCountPrefixSum;
    void *d_temp_storage;
    size_t temp_storage_bytes;

    // for marking and filtering particles
    int *d_valid;
    int *d_particleIds;
    int *d_cellKeys;
    int *d_cellCountPrefSumCopy;

    int *d_sortedCells;     // indices 0..NX*NY*NZ-1 sorted by descending cellCount
    int *d_cellCountSorted; // scratch — counts reordered alongside keys

    unsigned int *d_workQueueHead;

    // counters
    int sampleSteps;
    int NP;
    int *d_new_NP;  // allocated in device memory (GPU) as a temporary variable to be used

    unsigned long long totalCollisions;
    unsigned long long *d_totalCollisions;
} Simulation;

void setup(Simulation *sim, Config *conf);
void filter_and_index_particles(Simulation *sim, MPIHelper *mpiHelper);
void initialize_particles(Simulation *sim, MPIHelper *mpiHelper);
void apply_boundary_conditions_free_stream(Simulation *sim, MPIHelper *mpiHelper);
void move_particles(Simulation *sim);
void accumulate_sampling(Simulation *sim, MPIHelper *mpiHelper);
void clearPointers(Simulation *sim);

#endif // SIMULATION_H