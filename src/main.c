#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "../include/collider.h"
#include "../include/simulation.h"
#include "../include/io_utils.h"


int main(void) {
    srand((unsigned int)time(NULL));
    
    Simulation sim;
    Config conf;

    #ifdef WING_CASE
        printf("Running Wing case.\n");
    #elif defined(BALL_CASE)
        printf("Running Ball case.\n");
    #else
        print("No case selected. Aborting.\n");
        return 0;
    #endif

    config_setup(&conf);
    setup(&sim, &conf);
    initialize_particles(&sim, &conf);

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

    print_global_diagnostics(&sim, &conf, conf.nSteps);
    write_averaged_macros(&sim, &conf, "fields_avg.dat");
    write_paraview_files(&sim, &conf, conf.nSteps);

    clearPointers(&sim);

    return 0;
}
