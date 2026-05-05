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

    #ifdef WING_CASE
        printf("Running Wing case.\n");
    #elif defined(BALL_CASE)
        printf("Running Ball case.\n");
    #else
        print("No case selected. Aborting.\n");
        return 0;
    #endif

    setup(&sim);
    collider_setup();
    initialize_particles(&sim);

    for (int step = 0; step < nSteps; step++) {
        move_particles(&sim);
        apply_boundary_conditions_free_stream(&sim);
        index_particles(&sim);
        
        sim.totalCollisions += 
            collide_particles(sim.P, sim.cellCount, sim.cellList, sim.weight, sim.cellVolume);

        if (step >= firstSampleStep && step % samplingPeriod == 0) {
            accumulate_sampling(&sim);
        }

        if (step % printPeriod == 0) {
            print_global_diagnostics(&sim, step);
        }
    }

    print_global_diagnostics(&sim, nSteps);
    write_averaged_macros(&sim, "fields_avg.dat");
    write_paraview_files(&sim, nSteps);

    clearPointers(&sim);

    return 0;
}
