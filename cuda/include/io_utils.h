#ifndef IO_UTILS_H
#define IO_UTILS_H

#include "simulation.h"

void print_global_diagnostics(Simulation *sim, int step);
void write_averaged_macros(Simulation *sim, const char *filename, Cell *global_samples);
void write_paraview_files(Simulation *sim, unsigned int step, Cell *global_samples);
void move_necessary_data_before_printing(Simulation *sim);

#endif // IO_UTILS_H
