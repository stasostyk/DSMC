#include "../include/collider.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/particles.h"
#include "../include/simulation.h"
#include "../include/cuda_utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ bool collide_pair(Particles P, int i, int j, float multiplier, curandState *rngState) {
    float vx_i = P.vel[IDX_PARTICLE(i, 0)];
    float vy_i = P.vel[IDX_PARTICLE(i, 1)];
    float vz_i = P.vel[IDX_PARTICLE(i, 2)];

    float vx_j = P.vel[IDX_PARTICLE(j, 0)];
    float vy_j = P.vel[IDX_PARTICLE(j, 1)];
    float vz_j = P.vel[IDX_PARTICLE(j, 2)];

    float dvx = vx_j - vx_i;
    float dvy = vy_j - vy_i;
    float dvz = vz_j - vz_i;
    float relativeSpeed = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);

    float collisionProbSquared = multiplier * relativeSpeed;
    float u = curand_uniform(rngState);
    if (u*u < collisionProbSquared) {
        float N[3];
        random_isotropic_vector_device(N, rngState);
        relativeSpeed *= 0.5f;
        float VC[3] = { 0.5f * ( vx_j + vx_i ), 0.5f * ( vy_j + vy_i ), 0.5f * ( vz_j + vz_i ) };
        float VCr[3] = { relativeSpeed * N[0], relativeSpeed * N[1], relativeSpeed * N[2] };
        P.vel[IDX_PARTICLE(i, 0)] = VC[0] + VCr[0];
        P.vel[IDX_PARTICLE(i, 1)] = VC[1] + VCr[1];
        P.vel[IDX_PARTICLE(i, 2)] = VC[2] + VCr[2];
        P.vel[IDX_PARTICLE(j, 0)] = VC[0] - VCr[0];
        P.vel[IDX_PARTICLE(j, 1)] = VC[1] - VCr[1];
        P.vel[IDX_PARTICLE(j, 2)] = VC[2] - VCr[2];

        return true;
    }
    return false;
}

__global__ void ntcs_work_queue_kernel(
    unsigned long long *total_collisions,
    Particles P, int *cellCount, int *cellCountPrefixSum,
    curandState *rngStates, int *sortedCells, int totalCells,
    unsigned int *workQueueHead
) {
    __shared__ unsigned int collisionsBlock[64];
    unsigned int collisions = 0;

    int true_tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState rngState = rngStates[true_tid];

    // Each thread pulls work items until queue is empty
    while (true) {
        // Grab next cell from queue — one cell per thread
        unsigned int cellQueueIdx = atomicAdd(workQueueHead, 1);

        if (cellQueueIdx >= totalCells) break;

        // Use pre-sorted order: heavy cells first
        int idx = sortedCells[cellQueueIdx];
        int NPC = cellCount[idx];
        // if (NPC < 2 || NPC >= d_conf.hss_threshold) continue; 
        // HSS threshold doesn't need to be checked because we moved the queue head already.
        if (NPC < 2) continue;

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

            if (collide_pair(P, i, j, d_conf.ntcs_collisionProbMultiplierSquared, &rngState)) {
                collisions += 1;
            }

        }
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

__global__ void find_queue_start_kernel(
    unsigned int *workQueueHead,
    const int *sortedCells,
    const int *cellCount,
    int totalCells
) {
    // Single-thread binary search on device
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int lo = 0, hi = totalCells;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cellCount[sortedCells[mid]] >= d_conf.hss_threshold)
            lo = mid + 1;
        else
            hi = mid;
    }
    *workQueueHead = (unsigned int)lo;
}

__global__ void hss_scheme_kernel(unsigned long long *total_collisions,
                                  Particles P,
                                  int *cellCount,
                                  int *cellCountPrefixSum,
                                  curandState *rngStates,
                                  int *sortedCells
                                ) {
    __shared__ unsigned int collisionsBlock[64];
    __shared__ int localParticleList[MAX_PARTICLES_PER_CELL];

    int blockSize = blockDim.x;
    int tid = threadIdx.x;

    unsigned int collisions = 0;

    int cell_idx = sortedCells[blockIdx.x];

    int NPC = cellCount[cell_idx];
    int N_x = (NPC % 2 == 0) ? NPC - 1 : NPC;
    int particle_id_offset = cellCountPrefixSum[cell_idx];

    int rngIdx = cell_idx * blockSize + tid;
    curandState rngState = rngStates[rngIdx];

    int nPairs = NPC / 2;
    int offset = (NPC + 1) / 2;

    for (int i = tid; i < NPC; i += blockSize) {
        localParticleList[i] = particle_id_offset + i;
    }
    __syncthreads();
    
    for (int b = 0; b < d_conf.hss_nbatch; b++) {
        // fisher-yates shuffle (could be optimized with parallel sorting of random keys)
        if (threadIdx.x == 0) {
            for (int i = 0; i < NPC; i++) {
                int j = i + (int) (curand_uniform(&rngState) * (NPC - i));
                if (j < NPC) {
                    int temp = localParticleList[i];
                    localParticleList[i] = localParticleList[j];
                    localParticleList[j] = temp;
                }
            }
        }
        __syncthreads();

        for (int i = tid; i < nPairs; i += blockSize) {
            int j = i + offset;
            if (j < NPC) {
                int i_global = localParticleList[i];
                int j_global = localParticleList[j];

                if (collide_pair(P, i_global, j_global, d_conf.hss_collisionProbMultiplierSquared * N_x * N_x, &rngState)) {
                    collisions += 1;
                }
            }

        }
        __syncthreads();
    }
    rngStates[rngIdx] = rngState;

    collisionsBlock[tid] = collisions;

    __syncthreads();

    // Sum all collisions from the threads in this block (reduction)
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
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
    // Binary search the queue head - done on device because cell counts are stored there.
    find_queue_start_kernel<<<1, 1>>>(
        sim->d_workQueueHead, sim->d_sortedCells, sim->d_cellCount, sim->conf->totalCells
    );
    CHECK_KERNELCALL();
    
    // move workQueueHead to host
    unsigned int h_workQueueHead;
    cudaMemcpy(&h_workQueueHead, sim->d_workQueueHead, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // launch hss scheme with as many blocks as cells above the threshold
    if (h_workQueueHead > 0) {
        hss_scheme_kernel<<<h_workQueueHead, 64>>>(
            sim->d_totalCollisions, sim->d_P,
            sim->d_cellCount, sim->d_cellCountPrefixSum,
            sim->rngStates, sim->d_sortedCells
        );
        CHECK_KERNELCALL();
    }
    

    // Launch exactly as many threads as the GPU can run simultaneously —
    // more than this just adds atomic contention on workQueueHead
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int threadsPerBlock = 64;
    int blocks = numSMs;

    ntcs_work_queue_kernel<<<blocks, threadsPerBlock>>>(
        sim->d_totalCollisions, sim->d_P,
        sim->d_cellCount, sim->d_cellCountPrefixSum,
        sim->rngStates, sim->d_sortedCells, NX*NY*NZ,
        sim->d_workQueueHead
    );
    CHECK_KERNELCALL();
}
