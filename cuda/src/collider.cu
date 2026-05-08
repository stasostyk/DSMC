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
void elastic_collision(Particle *p1, Particle *p2, double Cr, curandStatePhilox4_32_10_t *rngState) {
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
    int *total_collisions, Particle *P, int *cellCount, int *cellList, curandStatePhilox4_32_10_t *rngStates
) {
    __shared__ int collisionsBlock[64];

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z * blockDim.z + threadIdx.z;

    int collisions = 0;

    if (k < NX && l < NY && m < NZ) {
        int idx = IDX_CELL(k, l, m);
        int NPC = cellCount[idx];
        if (NPC >= 2) {

            int *IPC = &cellList[IDX_LIST(k, l, m, 0)];

            curandStatePhilox4_32_10_t rngState = rngStates[idx];

            float4 r = curand_uniform4(&rngState);

            // estimate number of collisions
            double estimatedCollidingPairs = NPC * (NPC - 1) * d_conf.ntcs_collidingPairsMultiplier;
            int expectedCollidingPairs = (int)estimatedCollidingPairs;
            if (r.x < estimatedCollidingPairs - expectedCollidingPairs) expectedCollidingPairs++;

            // monte carlo accept/reject pairs and collide
            int i, j; // two particles to collide

            for (int k = 0; k < expectedCollidingPairs; k++) {
                i = (int)(r.y * NPC);
                j = (i + 1 + ((int)(r.z * (NPC - 1)))) % NPC;

                i = IPC[i];
                j = IPC[j];

                double relativeVel[3] = { P[j].vx - P[i].vx, P[j].vy - P[i].vy, P[j].vz - P[i].vz };
                double relativeSpeed = sqrt(relativeVel[0] * relativeVel[0]
                                            + relativeVel[1] * relativeVel[1]
                                            + relativeVel[2] * relativeVel[2]);

                // The real value to be calculated:
                // double collisionProb = d_conf.ntcs_collisionProbMultiplier * pow(1.0 / relativeSpeed, d_conf.ntcs_collisionProbExponent) * relativeSpeed;
                // But, we assume omega=0.75, which lets us remove pow() in a simple way:
                double collisionProb = d_conf.ntcs_collisionProbMultiplier * sqrt(relativeSpeed);
                if (r.w < collisionProb) {
                    elastic_collision( &P[i], &P[j], relativeSpeed, &rngState );
                    collisions++;
                }

            }
            rngStates[idx] = rngState;
        }
    }

    // TODO maybe this calculation should be done in row major ordening?
    // (check warp convergence, and how cuda assigns IDs to 3d thread blocks)
    int tid =
        threadIdx.x * blockDim.y * blockDim.z +
        threadIdx.y * blockDim.z +
        threadIdx.z;
    collisionsBlock[tid] = collisions;

    __syncthreads();

    // Sum all collisions from the threads in this block (reduction)
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            collisionsBlock[tid] += collisionsBlock[tid + stride];
        }

        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(total_collisions, collisionsBlock[0]);
    }
}

int collide_particles(Simulation *sim) {
    dim3 threadsPerBlock(4, 4, 4);
    dim3 blocksPerGrid(
        (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    int *d_total_collisions;
    CHECK(cudaMalloc(&d_total_collisions, sizeof(int)));
    CHECK(cudaMemset(d_total_collisions, 0, sizeof(int)));

    no_time_counter_scheme_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_total_collisions, sim->d_P, sim->d_cellCount, sim->d_cellList, sim->rngStates
    );
    CHECK_KERNELCALL();

    int totalCollisions;
    CHECK(cudaMemcpy(&totalCollisions, d_total_collisions, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_total_collisions));

    return totalCollisions;
}
