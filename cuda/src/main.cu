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
//    CHECK(cudaMemcpy(sim->P, sim->d_P, sim->NP * 6 * sizeof(double), cudaMemcpyDeviceToHost));
//
    CHECK(cudaMemcpy(sim->P.x, sim->d_P.x, sim->NP * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.y, sim->d_P.y, sim->NP * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.z, sim->d_P.z, sim->NP * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.vx, sim->d_P.vx, sim->NP * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.vy, sim->d_P.vy, sim->NP * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.vz, sim->d_P.vz, sim->NP * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(sim->samples, sim->d_samples, SAMPLES_SZ, cudaMemcpyDeviceToHost));
} 

int main(void) {
    Timer t, allProgramTimer;

    timer_start(&allProgramTimer);
    timer_start(&t);

    srand((unsigned int)time(NULL));
    
    Simulation sim;
    Config conf;

    #ifdef WING_CASE
        printf("Running Wing case.\n");
    #elif defined(BALL_CASE)
        printf("Running Ball case.\n");
    #else
        printf("No case selected. Aborting.\n");
        return 0;
    #endif

    config_setup(&conf);
    setup(&sim, &conf);
    initialize_particles(&sim);

    timer_end(&t);
    timer_print(&t, "INITIALIZATION");
    timer_start(&t);

    for (int step = 0; step < conf.nSteps; step++) {
        move_particles(&sim);
        apply_boundary_conditions_free_stream(&sim);
        index_particles(&sim);
        
        collide_particles_hss(&sim);
//        collide_particles(&sim);

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
