#include "../include/collider.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/particle.h"
#include "../include/simulation.h"
#include "../include/cuda_utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__
void elastic_collision(Particle *p1, Particle *p2, double Cr, curandState *rngState) {
    double N[3];
    random_isotropic_vector_device(N, rngState);
    Cr *= 0.5;
    double VC[3] = { 0.5 * ( p2->vx + p1->vx ), 0.5 * ( p2->vy + p1->vy ), 0.5 * ( p2->vz + p1->vz ) };
    double VCr[3] = { Cr * N[0], Cr * N[1], Cr * N[2] };
    p1->vx = VC[0] + VCr[0];
    p1->vy = VC[1] + VCr[1];
    p1->vz = VC[2] + VCr[2];
    p2->vx = VC[0] - VCr[0];
    p2->vy = VC[1] - VCr[1];
    p2->vz = VC[2] - VCr[2];
}

__global__ void no_time_counter_scheme_kernel(
    int *total_collisions, Particle *P, int *cellCount, int *cellList,
    double weight, double cellVolume,
    Config *conf,
    curandState *rngStates
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = IDX_CELL(k, l, m);

    if (k >= NX || l >= NY || m >= NZ) return;

    int NPC = cellCount[IDX_CELL(k, l, m)];
    if (NPC < 2) return;

    int *IPC = &cellList[IDX_LIST(k, l, m, 0)];

    // TODO these prob can be moved to constant memory or #define
    // estimate number of collisions
    double majorant = 9.0 * conf->sigmaRef * sqrt(conf->KB * conf->TFree / conf->moleculeMass);
    double estimatedCollidingPairs = 0.5 * NPC * (NPC - 1) * weight * majorant * conf->dt / cellVolume;
    int expectedCollodingPairs = (int)estimatedCollidingPairs;
    if (curand_uniform(&rngStates[idx]) < estimatedCollidingPairs - expectedCollodingPairs) expectedCollodingPairs++;

    // monte carlo accept/reject pairs and collide
    int collisions = 0;
    int i, j; // two particles to collide

    for (int k = 0; k < expectedCollodingPairs; k++) {
        i = (int)(curand_uniform(&rngStates[idx]) * NPC);
        do {
            j = (int)(curand_uniform(&rngStates[idx]) * NPC);
        } while (j == i);

        i = IPC[i];
        j = IPC[j];

        double relativeVel[3] = { P[j].vx - P[i].vx, P[j].vy - P[i].vy, P[j].vz - P[i].vz };
        double relativeSpeed = sqrt(relativeVel[0] * relativeVel[0]
                                    + relativeVel[1] * relativeVel[1]
                                    + relativeVel[2] * relativeVel[2]);

        double collisionProb = conf->sigmaRef * pow(conf->CrRef / relativeSpeed, 2.0*conf->omega - 1.0) * relativeSpeed / majorant;
        if (curand_uniform(&rngStates[idx]) < collisionProb) {
            elastic_collision( &P[i], &P[j], relativeSpeed, &rngStates[idx] );
            collisions++;
        }

    }

    atomicAdd(total_collisions, collisions);
}

int collide_particles(
    Simulation *sim,
    Config *conf,
    double weight, 
    double cellVolume
) {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid(
        (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    int *d_total_collisions;
    CHECK(cudaMalloc(&d_total_collisions, sizeof(int)));

    Config *d_conf;
    CHECK(cudaMalloc(&d_conf, sizeof(Config)));

    CHECK(cudaMemcpy(sim->d_P, sim->P, PARTICLES_SZ, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sim->d_cellCount, sim->cellCount, CELL_COUNT_SZ, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sim->d_cellList, sim->cellList, CELL_LIST_SZ, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_conf, conf, sizeof(Config), cudaMemcpyHostToDevice));

    no_time_counter_scheme_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_total_collisions, sim->d_P, sim->d_cellCount, sim->d_cellList,
        weight, cellVolume, sim->rngStates
    );
    CHECK_KERNELCALL();

    int totalCollisions;
    CHECK(cudaMemcpy(&totalCollisions, d_total_collisions, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_total_collisions));

    CHECK(cudaMemcpy(sim->P, sim->d_P, PARTICLES_SZ, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->cellCount, sim->d_cellCount, CELL_COUNT_SZ, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->cellList, sim->d_cellList, CELL_LIST_SZ, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_conf));

    return totalCollisions;
}
