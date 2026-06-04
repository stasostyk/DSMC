#ifndef MPI_EXCHANGE_H
#define MPI_EXCHANGE_H

#include "mpi_helper.h"
#include "simulation.h"
#include "config.h"

void exchange_boundary_particles(Simulation *sim, MPIHelper *mpiHelper);
void setupMPIHelper(MPIHelper *mpiHelper, Config *conf);

#endif