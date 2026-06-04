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
#include "../include/mpi_helper.h"
#include "../include/mpi_exchange.h"

#include <mpi.h>

void move_neccessary_data_before_printing(Simulation *sim) {
    // Particle data is mostly stored in and dealt in GPU, 
    // to have the newest version in CPU, it needs to be copied.
   
    CHECK(cudaMemcpy(&sim->totalCollisions, sim->d_totalCollisions, sizeof(unsigned long long), cudaMemcpyDeviceToHost))

    CHECK(cudaMemcpy(sim->P.x, sim->d_P.x, sim->NP * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.y, sim->d_P.y, sim->NP * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.z, sim->d_P.z, sim->NP * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.vx, sim->d_P.vx, sim->NP * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.vy, sim->d_P.vy, sim->NP * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->P.vz, sim->d_P.vz, sim->NP * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(sim->samples, sim->d_samples, SAMPLES_SZ, cudaMemcpyDeviceToHost));
} 

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
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
    MPIHelper mpiHelper;

    config_setup(&conf, object_case);
    setupMPIHelper(&mpiHelper, &conf);
    setup(&sim, &conf);

    if (NX % mpiHelper.worldSize != 0) {
        printf("NX SHOULD DIVIDE BY MPI NODE COUNT!\n");
        MPI_Finalize();
        return 1;
    }

    initialize_particles(&sim, &mpiHelper);

    timer_end(&t);
    timer_print(&t, "INITIALIZATION");
    timer_start(&t);

    for (int step = 0; step < conf.nSteps; step++) {
        move_particles(&sim);
        apply_boundary_conditions_free_stream(&sim, &mpiHelper);

        exchange_boundary_particles(&sim, &mpiHelper);

        filter_and_index_particles(&sim, true);
        
        // collide_particles_hss(&sim);
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

    // Each rank has samples for its local cells
    // Reduce to rank 0 for output
    // int world_size = mpiHelper->worldSize;

    Cell *global_samples = NULL;
    if (mpiHelper.worldRank == 0) {
        global_samples = (Cell *)malloc(SAMPLES_SZ);
        memset(global_samples, 0, SAMPLES_SZ);
    }

    // Each rank has samples for its local cells
    // Reduce to rank 0 for output
    MPI_Reduce(sim.samples, global_samples,
                sizeof(Cell)/sizeof(float) * NX*NY*NZ,
                MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpiHelper.worldRank == 0) {
        print_global_diagnostics(&sim, conf.nSteps);
        write_averaged_macros(&sim, "fields_avg.dat", global_samples);
        if (global_samples != NULL) free(global_samples);
        // write_paraview_files(&sim, conf.nSteps);
    }

    clearPointers(&sim);

    timer_end(&allProgramTimer);
    timer_print(&allProgramTimer, "ALL PROGRAM FINISHED");

    CHECK(cudaFree(mpiHelper.d_prefix_left));
    CHECK(cudaFree(mpiHelper.d_prefix_right));
    CHECK(cudaFree(mpiHelper.d_count));
    CHECK(cudaFree(mpiHelper.d_send_left));
    CHECK(cudaFree(mpiHelper.d_send_right));
    CHECK(cudaFreeHost(mpiHelper.h_recv_left));
    CHECK(cudaFreeHost(mpiHelper.h_recv_right));
    CHECK(cudaFreeHost(mpiHelper.h_send_left));
    CHECK(cudaFreeHost(mpiHelper.h_send_right));

    MPI_Finalize();

    return 0;
}
