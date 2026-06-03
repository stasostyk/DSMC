#ifndef IO_UTILS_H
#define IO_UTILS_H

#include "simulation.h"
#include "mpi_helper.h"

void print_global_diagnostics(Simulation *sim, int step);
void write_averaged_macros(Simulation *sim, const char *filename, MPIHelper *mpiHelper);
void write_paraview_files(Simulation *sim, unsigned int step);

#endif // IO_UTILS_H
