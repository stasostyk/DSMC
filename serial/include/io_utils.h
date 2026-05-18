#ifndef IO_UTILS_H
#define IO_UTILS_H

#include "simulation.h"

void print_global_diagnostics(Simulation *sim, Config *conf, int step);
void write_averaged_macros(Simulation *sim, Config *conf, const char *filename);
void write_vti(Simulation *sim, Config *conf, const char *filename);

#ifdef WING_CASE
void write_wing_vtp(Config *conf, const char *filename);
#elif defined(BALL_CASE)
void write_ball_vtp(Config *conf, const char *filename);
#endif

void write_paraview_files(Simulation *sim, Config *conf, unsigned int step);

#endif // IO_UTILS_H
