#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "../include/collider.h"
#include "../include/simulation.h"
#include "../include/io_utils.h"
#include "../include/timer.h"


int main(void) {
    Timer t, allProgramTimer;

    timer_start(&allProgramTimer);
    timer_start(&t);

    srand((unsigned int)time(NULL));
    
    Simulation sim;
    Config conf;

    #ifdef WING_CASE
        printf("Running Wing case.\n");
    #elif defined(SPHERE_CASE)
        printf("Running Sphere case.\n");
    #else
        print("No case selected. Aborting.\n");
        return 0;
    #endif

    config_setup(&conf);
    setup(&sim, &conf);
    initialize_particles(&sim, &conf);

    timer_end(&t);
    timer_print(&t, "INITIALIZATION");
    timer_start(&t);

    for (int step = 0; step < conf.nSteps; step++) {
        move_particles(&sim, &conf);
        apply_boundary_conditions_free_stream(&sim, &conf);
        index_particles(&sim);
        
        sim.totalCollisions += 
            collide_particles(&conf, sim.P, sim.cellCount, sim.cellList, sim.weight, sim.cellVolume);

        if (step >= conf.firstSampleStep && step % conf.samplingPeriod == 0) {
            accumulate_sampling(&sim);
        }

        if (step % conf.printPeriod == 0) {
            print_global_diagnostics(&sim, &conf, step);
        }
    }

    timer_end(&t);
    timer_print(&t, "SIMULATION LOOP");

    print_global_diagnostics(&sim, &conf, conf.nSteps);
    write_averaged_macros(&sim, &conf, "fields_avg.dat");
    write_paraview_files(&sim, &conf, conf.nSteps);

    clearPointers(&sim);

    timer_end(&allProgramTimer);
    timer_print(&allProgramTimer, "ALL PROGRAM FINISHED");

    return 0;
}
