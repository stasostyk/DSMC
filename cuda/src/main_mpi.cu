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
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Assign one GPU per MPI rank
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    CHECK(cudaSetDevice(world_rank % num_gpus));

    MPIHelper mpiHelper;
    mpiHelper.worldRank = world_rank;
    mpiHelper.worldSize = world_size;
    mpiHelper.numGpus = num_gpus;

    if (NX % mpiHelper.worldSize != 0) {
        printf("NX SHOULD DIVIDE BY MPI NODE COUNT!\n");
        MPI_Finalize();
        return 0;
    }

    printf("world rank: %d\n", world_rank);
    printf("world size: %d\n", world_size);
    printf("num gpus: %d\n", num_gpus);

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

    mpiHelper.slabWidth = conf.Lx / (float)world_size;
    mpiHelper.xMin = (float)world_rank * mpiHelper.slabWidth;
    mpiHelper.xMax = mpiHelper.xMin + mpiHelper.slabWidth;

    mpiHelper.left_rank = (world_rank > 0) ? world_rank - 1 : MPI_PROC_NULL;
    mpiHelper.right_rank = (world_rank < world_size-1) ? world_rank + 1 : MPI_PROC_NULL;


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, world_rank % num_gpus);
    if (!prop.canMapHostMemory) {
        fprintf(stderr, "Device does NOT support mapped host memory\n");
        MPI_Finalize();
        return 0;
    }
    // mpiHelper.kOffset = world_rank * NX / world_size;
    // mpiHelper.kOffset = 0;

    printf("slab width: %f\n", mpiHelper.slabWidth);
    printf("xmin: %f\n", mpiHelper.xMin);
    printf("xmax: %f\n", mpiHelper.xMax);
    // printf("kOffset: %d\n", mpiHelper.kOffset);

    setup(&sim, &conf);

    // Assume (because of the stream), more particles going right
    // Values got empirically
    int particlesGoingRight = MAX_PARTICLES / 10;
    int particlesGoingLeft = MAX_PARTICLES / 10;

    mpiHelper.bufferToSendLeftCount = particlesGoingLeft;
    mpiHelper.bufferToSendRightCount = particlesGoingRight;

    CHECK(cudaMalloc(&mpiHelper.d_count,  2 * sizeof(int)));
    CHECK(cudaMalloc(&mpiHelper.d_send_left, 6 * sizeof(float) * particlesGoingLeft));
    CHECK(cudaMalloc(&mpiHelper.d_send_right, 6 * sizeof(float) * particlesGoingRight));
    // CHECK(cudaMalloc(&mpiHelper.d_recv_left, 6 * sizeof(float) * smallerParticleSize));
    // CHECK(cudaMalloc(&mpiHelper.d_recv_right, 6 * sizeof(float) * smallerParticleSize));
    // CHECK(cudaMalloc(&mpiHelper.d_flag, sizeof(int) * MAX_PARTICLES));
    // CHECK(cudaMalloc(&mpiHelper.d_prefix_keep, sizeof(int) * MAX_PARTICLES));
    CHECK(cudaMalloc(&mpiHelper.d_prefix_right, sizeof(int) * MAX_PARTICLES));
    CHECK(cudaMalloc(&mpiHelper.d_prefix_left, sizeof(int) * MAX_PARTICLES));

    // Use pinned memory for faster D2H/H2D transfers
    CHECK(cudaMallocHost(&mpiHelper.h_send_left,  particlesGoingLeft * 6 * sizeof(float)));
    CHECK(cudaMallocHost(&mpiHelper.h_send_right, particlesGoingRight * 6 * sizeof(float)));
    CHECK(cudaMallocHost(&mpiHelper.h_recv_left,  particlesGoingRight * 6 * sizeof(float)));
    CHECK(cudaMallocHost(&mpiHelper.h_recv_right, particlesGoingLeft * 6 * sizeof(float)));

    // Recv side: access is sequential in unpack_recv_kernel, so zero-copy is fine
    cudaHostAlloc(&mpiHelper.h_recv_left,  particlesGoingRight * 6 * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&mpiHelper.h_recv_right, particlesGoingLeft * 6 * sizeof(float), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&mpiHelper.d_recv_left_mapped,  mpiHelper.h_recv_left,  0);
    cudaHostGetDevicePointer(&mpiHelper.d_recv_right_mapped, mpiHelper.h_recv_right, 0);


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
    if (world_rank == 0) {
        global_samples = (Cell *)malloc(SAMPLES_SZ);
        memset(global_samples, 0, SAMPLES_SZ);
    }

    // Each rank has samples for its local cells
    // Reduce to rank 0 for output
    MPI_Reduce(sim.samples, global_samples,
                sizeof(Cell)/sizeof(float) * NX*NY*NZ,
                MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        print_global_diagnostics(&sim, conf.nSteps);
        write_averaged_macros(&sim, "fields_avg.dat", global_samples);
        if (global_samples != NULL) free(global_samples);
        // write_paraview_files(&sim, conf.nSteps);
    }

    clearPointers(&sim);

    timer_end(&allProgramTimer);
    timer_print(&allProgramTimer, "ALL PROGRAM FINISHED");

    // CHECK(cudaFree(mpiHelper.d_flag));
    // CHECK(cudaFree(mpiHelper.d_prefix_keep));
    CHECK(cudaFree(mpiHelper.d_prefix_left));
    CHECK(cudaFree(mpiHelper.d_prefix_right));
    CHECK(cudaFree(mpiHelper.d_count));
    CHECK(cudaFree(mpiHelper.d_send_left));
    CHECK(cudaFree(mpiHelper.d_send_right));
    // CHECK(cudaFree(mpiHelper.d_recv_left));
    // CHECK(cudaFree(mpiHelper.d_recv_right));
    CHECK(cudaFreeHost(mpiHelper.h_recv_left));
    CHECK(cudaFreeHost(mpiHelper.h_recv_right));
    CHECK(cudaFreeHost(mpiHelper.h_send_left));
    CHECK(cudaFreeHost(mpiHelper.h_send_right));

    MPI_Finalize();

    return 0;
}
