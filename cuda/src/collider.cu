#include "../include/collider.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/particles.h"
#include "../include/simulation.h"
#include "../include/cuda_utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__
void elastic_collision(double *vx1,
                       double *vy1,
                       double *vz1,
                       double *vx2,
                       double *vy2,
                       double *vz2,
                       double Cr,
                       curandState *rngState) {
    double N[3];
    random_isotropic_vector_device(N, rngState);
    Cr *= 0.5;
    double VC[3] = { 0.5 * ( *vx2 + *vx1 ), 0.5 * ( *vy2 + *vy1 ), 0.5 * ( *vz2 + *vz1 ) };
    double VCr[3] = { Cr * N[0], Cr * N[1], Cr * N[2] };
    *vx1 = VC[0] + VCr[0];
    *vy1 = VC[1] + VCr[1];
    *vz1 = VC[2] + VCr[2];
    *vx2 = VC[0] - VCr[0];
    *vy2 = VC[1] - VCr[1];
    *vz2 = VC[2] - VCr[2];
}

__global__ void no_time_counter_scheme_kernel(
    unsigned long long *total_collisions, Particles P, int *cellCount, int *cellList, curandState *rngStates
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

            curandState rngState = rngStates[idx];

            // estimate number of collisions
            double estimatedCollidingPairs = NPC * (NPC - 1) * d_conf.ntcs_collidingPairsMultiplier;
            int expectedCollidingPairs = (int)estimatedCollidingPairs;
            if (curand_uniform(&rngState) < estimatedCollidingPairs - expectedCollidingPairs) expectedCollidingPairs++;

            // monte carlo accept/reject pairs and collide

            for (int k = 0; k < expectedCollidingPairs; k++) {
                int i_local = (int)(curand_uniform(&rngState) * NPC);
                int j_offset = (int)(curand_uniform(&rngState) * (NPC - 1));
                int j_local = (i_local + 1 + j_offset) % NPC;

                
                // two particles to collide
                int i = IPC[i_local];
                int j = IPC[j_local];

                double relativeVel[3] = { P.vx[j] - P.vx[i], P.vy[j] - P.vy[i], P.vz[j] - P.vz[i] };
                double relativeSpeed = sqrt(relativeVel[0] * relativeVel[0]
                                            + relativeVel[1] * relativeVel[1]
                                            + relativeVel[2] * relativeVel[2]);

                // The real value to be calculated:
                // double collisionProb = d_conf.ntcs_collisionProbMultiplier * pow(1.0 / relativeSpeed, d_conf.ntcs_collisionProbExponent) * relativeSpeed;
                // But, we assume omega=0.75, which lets us remove pow() in a simple way:
                double collisionProb = d_conf.ntcs_collisionProbMultiplier * sqrt(relativeSpeed);
                if (curand_uniform(&rngState) < collisionProb) {
                    elastic_collision(&P.vx[i],
                                      &P.vy[i],
                                      &P.vz[i],
                                      &P.vx[j],
                                      &P.vy[j],
                                      &P.vz[j],
                                      relativeSpeed,
                                      &rngState );
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
        atomicAdd(total_collisions, (unsigned long long)collisionsBlock[0]);
    }
}

void collide_particles(Simulation *sim) {
    dim3 threadsPerBlock(4, 4, 4);
    dim3 blocksPerGrid(
        (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    no_time_counter_scheme_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_totalCollisions, sim->d_P, sim->d_cellCount, sim->d_cellList, sim->rngStates
    );
    CHECK_KERNELCALL();
}
