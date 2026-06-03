#include "../include/collider.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/particles.h"
#include "../include/simulation.h"
#include "../include/cuda_utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void no_time_counter_scheme_kernel(
    unsigned long long *total_collisions, 
    Particles P, int *cellCount, int *cellCountPrefixSum, curandState *rngStates
) {
    __shared__ unsigned int collisionsBlock[64];

    // x direction varies fastest, and IDX_LIST and IDX_CELL are stored row major
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int collisions = 0;

    if (k < NX && l < NY && m < NZ) {
        int idx = IDX_CELL(k, l, m);
        int NPC = cellCount[idx];
        if (NPC >= 2 && NPC < d_conf.hss_threshold) {

            // Particles for this cell are contiguous starting at this offset
            int offset = cellCountPrefixSum[idx];

            curandState rngState = rngStates[idx];

            // estimate number of collisions
            float estimatedCollidingPairs = NPC * (NPC - 1) * d_conf.ntcs_collidingPairsMultiplier;
            int expectedCollidingPairs = (int)estimatedCollidingPairs;
            if (curand_uniform(&rngState) < estimatedCollidingPairs - expectedCollidingPairs) expectedCollidingPairs++;

            // monte carlo accept/reject pairs and collide

            for (int k = 0; k < expectedCollidingPairs; k++) {
                int i_local = (int)(curand_uniform(&rngState) * NPC);
                int j_offset = (int)(curand_uniform(&rngState) * (NPC - 1));
                int j_local = (i_local + 1 + j_offset) % NPC;

                // two particles to collide
                int i = i_local + offset;
                int j = j_local + offset;

                float vx_i = P.vel[IDX_PARTICLE(i, 0)];
                float vy_i = P.vel[IDX_PARTICLE(i, 1)];
                float vz_i = P.vel[IDX_PARTICLE(i, 2)];

                float vx_j = P.vel[IDX_PARTICLE(j, 0)];
                float vy_j = P.vel[IDX_PARTICLE(j, 1)];
                float vz_j = P.vel[IDX_PARTICLE(j, 2)];

                float dvx = vx_j - vx_i;
                float dvy = vy_j - vy_i;
                float dvz = vz_j - vz_i;
                float relativeSpeed = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);

                // The real value to be calculated:
                // double collisionProb = d_conf.ntcs_collisionProbMultiplier * pow(1.0 / relativeSpeed, d_conf.ntcs_collisionProbExponent) * relativeSpeed;
                // But, we assume omega=0.75, which lets us remove pow() in a simple way:
                float collisionProbSquared = d_conf.ntcs_collisionProbMultiplierSquared * relativeSpeed;
                float u = curand_uniform(&rngState);
                if (u*u < collisionProbSquared) {
                    float N[3];
                    random_isotropic_vector_device(N, &rngState);
                    relativeSpeed *= 0.5f;
                    float VC[3] = { 0.5f * ( vx_j + vx_i ), 0.5f * ( vy_j + vy_i ), 0.5f * ( vz_j + vz_i ) };
                    float VCr[3] = { relativeSpeed * N[0], relativeSpeed * N[1], relativeSpeed * N[2] };
                    P.vel[IDX_PARTICLE(i, 0)] = VC[0] + VCr[0];
                    P.vel[IDX_PARTICLE(i, 1)] = VC[1] + VCr[1];
                    P.vel[IDX_PARTICLE(i, 2)] = VC[2] + VCr[2];
                    P.vel[IDX_PARTICLE(j, 0)] = VC[0] - VCr[0];
                    P.vel[IDX_PARTICLE(j, 1)] = VC[1] - VCr[1];
                    P.vel[IDX_PARTICLE(j, 2)] = VC[2] - VCr[2];

                    collisions++;
                }

            }
            rngStates[idx] = rngState;
        }
    }

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
        (NZ + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NX + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    no_time_counter_scheme_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_totalCollisions, sim->d_P, 
        sim->d_cellCount, sim->d_cellCountPrefixSum, 
        sim->rngStates
    );
    CHECK_KERNELCALL();
}
