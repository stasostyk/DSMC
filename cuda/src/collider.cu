#include "../include/collider.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/particles.h"
#include "../include/simulation.h"
#include "../include/cuda_utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void ntcs_persistent_kernel(
    unsigned long long *total_collisions,
    Particles P, int *cellCount, int *cellCountPrefixSum,
    curandState *rngStates, int *sortedCells, int totalCells,
    unsigned int *workQueueHead
) {
    __shared__ unsigned int collisionsBlock[256];
    unsigned int collisions = 0;

    int true_tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState rngState = rngStates[true_tid];

    // Each thread pulls work items until queue is empty
    while (true) {
        // Grab next cell from queue — one cell per thread
        unsigned int cellQueueIdx = atomicAdd(workQueueHead, 1);

        // // Replaced the single atomicAdd with warp-aggregated version:
        // unsigned int cellQueueIdx;
        // if ((threadIdx.x % 32) == 0) {
        //     cellQueueIdx = atomicAdd(workQueueHead, 32);
        // }
        // // Broadcast from lane 0 to all lanes in warp
        // cellQueueIdx = __shfl_sync(0xFFFFFFFF, cellQueueIdx, 0);
        // cellQueueIdx += (threadIdx.x % 32);  // each lane gets its own index

        if (cellQueueIdx >= totalCells) break;

        // Use pre-sorted order: heavy cells first
        int idx = sortedCells[cellQueueIdx];
        int NPC = cellCount[idx];
        if (NPC < 2 || NPC >= d_conf.hss_threshold) continue;

        // Particles for this cell are contiguous starting at this offset
        int offset = cellCountPrefixSum[idx];

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

            float vx_i = P.vx[i];
            float vy_i = P.vy[i];
            float vz_i = P.vz[i];

            float vx_j = P.vx[j];
            float vy_j = P.vy[j];
            float vz_j = P.vz[j];

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
                P.vx[i] = VC[0] + VCr[0];
                P.vy[i] = VC[1] + VCr[1];
                P.vz[i] = VC[2] + VCr[2];
                P.vx[j] = VC[0] - VCr[0];
                P.vy[j] = VC[1] - VCr[1];
                P.vz[j] = VC[2] - VCr[2];

                collisions++;
            }

        }

        // cellQueueIdx += 31 - (threadIdx.x % 32);  // each lane gets its own index

        // if (cellQueueIdx >= totalCells) break;
    }

    rngStates[true_tid] = rngState;

    // Reduction 
    int tid = threadIdx.x;
    collisionsBlock[tid] = collisions;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) collisionsBlock[tid] += collisionsBlock[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(total_collisions, (unsigned long long)collisionsBlock[0]);
}

void collide_particles(Simulation *sim) {
    cudaMemset(sim->d_workQueueHead, 0, sizeof(unsigned int));

    // Launch exactly as many threads as the GPU can run simultaneously —
    // more than this just adds atomic contention on workQueueHead
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int threadsPerBlock = 64;
    int blocks = numSMs * 2;  // 4 waves per SM keeps the queue hot

    ntcs_persistent_kernel<<<blocks, threadsPerBlock>>>(
        sim->d_totalCollisions, sim->d_P,
        sim->d_cellCount, sim->d_cellCountPrefixSum,
        sim->rngStates, sim->d_sortedCells, NX*NY*NZ,
        sim->d_workQueueHead
    );
    CHECK_KERNELCALL();

    // dim3 threadsPerBlock(4, 4, 4);
    // dim3 blocksPerGrid(
    //     (NZ + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //     (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
    //     (NX + threadsPerBlock.z - 1) / threadsPerBlock.z
    // );

    // no_time_counter_scheme_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //     sim->d_totalCollisions, sim->d_P, 
    //     sim->d_cellCount, sim->d_cellCountPrefixSum, 
    //     sim->rngStates
    // );
    // CHECK_KERNELCALL();
}
