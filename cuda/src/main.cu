#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/collider_hss.h"
#include "../include/collider.h"
#include "../include/simulation.h"
#include "../include/io_utils.h"
#include "../include/timer.h"
#include "../include/cuda_utils.h"

void move_neccessary_data_before_printing(Simulation *sim) {
    // Particle data is mostly stored in and dealt in GPU, 
    // to have the newest version in CPU, it needs to be copied.
   
    CHECK(cudaMemcpy(&sim->totalCollisions, sim->d_totalCollisions, sizeof(unsigned long long), cudaMemcpyDeviceToHost))

    CHECK(cudaMemcpy(sim->P.pos, sim->d_P.pos, sim->NP * sizeof(float) * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.vel, sim->d_P.vel, sim->NP * sizeof(float) * 4, cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(sim->samples, sim->d_samples, SAMPLES_SZ, cudaMemcpyDeviceToHost));
} 

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: ./DSMC [case], where case is \"BALL\" or \"WING\".\n");
        return 0;
    }

    int object_case;
    if (strcmp("BALL", argv[1]) == 0) {
        object_case = 0; 
    } else if (strcmp("WING", argv[1]) == 0) {
        object_case = 1;
    } else if (strcmp("COMBO", argv[1]) == 0) {
        object_case = 2;
    } else {
        printf("Usage: ./DSMC [case], where case is \"BALL\" or \"WING\".\n");
        printf("given case: %s", argv[1]);
        return 0;
    }
    printf("Running case: %s\n", argv[1]);

    Timer t, allProgramTimer;

    timer_start(&allProgramTimer);
    timer_start(&t);

    Simulation sim;
    Config conf;

    config_setup(&conf, object_case);
    setup(&sim, &conf);
    initialize_particles(&sim);

    timer_end(&t);
    timer_print(&t, "INITIALIZATION");
    timer_start(&t);

    for (int step = 0; step < conf.nSteps; step++) {
        move_particles(&sim);
        apply_boundary_conditions_free_stream(&sim);
        filter_and_index_particles(&sim);
        
        collide_particles_hss(&sim);
        collide_particles(&sim);

        if (step >= conf.firstSampleStep && step % conf.samplingPeriod == 0) {
            accumulate_sampling(&sim);
        }

        if (conf.printPeriod > 0 && step % conf.printPeriod == 0) {
            move_neccessary_data_before_printing(&sim);
            print_global_diagnostics(&sim, step);
        }
    }

    timer_end(&t);
    timer_print(&t, "SIMULATION LOOP");

    move_neccessary_data_before_printing(&sim);

    print_global_diagnostics(&sim, conf.nSteps);
    write_averaged_macros(&sim, "fields_avg.dat");
    write_paraview_files(&sim, conf.nSteps);

    clearPointers(&sim);

    timer_end(&allProgramTimer);
    timer_print(&allProgramTimer, "ALL PROGRAM FINISHED");

    return 0;
}
