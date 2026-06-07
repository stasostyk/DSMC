#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/collider.h"
#include "../include/simulation.h"
#include "../include/io_utils.h"
#include "../include/timer.h"
#include "../include/cuda_utils.h"
#include "../include/mpi_helper.h"
#include "../include/mpi_exchange.h"

#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    if (argc < 2) {
        printf("Usage: ./DSMC [case], where case is \"SPHERE\" or \"WING\".\n");
        return 0;
    }

    int object_case;
    if (strcmp("SPHERE", argv[1]) == 0) {
        object_case = 0; 
    } else if (strcmp("WING", argv[1]) == 0) {
        object_case = 1;
    } else if (strcmp("COMBO", argv[1]) == 0) {
        object_case = 2;
    } else {
        printf("Usage: ./DSMC [case], where case is \"SPHERE\" or \"WING\".\n");
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
        
        collide_particles(&sim);

        if (step >= conf.firstSampleStep && step % conf.samplingPeriod == 0) {
            accumulate_sampling(&sim);
        }

        if (conf.printPeriod > 0 && step % conf.printPeriod == 0) {
            move_necessary_data_before_printing(&sim);
            print_global_diagnostics(&sim, step);
        }
    }

    timer_end(&t);
    timer_print(&t, "SIMULATION LOOP");

    move_necessary_data_before_printing(&sim);

    Cell *global_samples = reduceSamples(&sim, &mpiHelper);

    if (mpiHelper.worldRank == 0) {
        print_global_diagnostics(&sim, conf.nSteps);
        write_averaged_macros(&sim, "fields_avg.dat", global_samples);
        write_paraview_files(&sim, conf.nSteps, global_samples);
    }

    if (global_samples != NULL) free(global_samples);

    clearPointers(&sim);

    timer_end(&allProgramTimer);
    timer_print(&allProgramTimer, "ALL PROGRAM FINISHED");

    MPI_Finalize();

    return 0;
}
