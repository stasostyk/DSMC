#include "../include/collider.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/particles.h"
#include "../include/simulation.h"
#include "../include/cuda_utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void child_hss_scheme_kernel(unsigned long long *total_collisions,
                                  Particles P,
                                  int NPC,
                                  int N_x,
                                  int *cellList,
                                  curandState *rngStates,
                                  int cell_idx
                                ) {
    __shared__ int collisionsBlock[64];
    __shared__ int localParticleList[MAX_PARTICLES_PER_CELL];

    int blockSize = blockDim.x;

    int tid = threadIdx.x;

    int collisions = 0;

    int rngIdx = cell_idx * blockSize + tid;
    curandState rngState = rngStates[rngIdx];

    int nPairs = NPC / 2;
    int offset = (NPC + 1) / 2;
    for (int i = tid; i < NPC; i += blockSize) {
        localParticleList[i] =
                cellList[cell_idx * MAX_PARTICLES_PER_CELL + i];
    }

    for (int b = 0; b < d_conf.hss_nbatch; b++) {
        __syncthreads();
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

                float vx_i = P.vx[i_global];
                float vy_i = P.vy[i_global];
                float vz_i = P.vz[i_global];

                float vx_j = P.vx[j_global];
                float vy_j = P.vy[j_global];
                float vz_j = P.vz[j_global];

                // float relativeVel[3] = {vx_j - vx_i, vy_j - vy_i, vz_j - vz_i};
                float relativeVel_x = vx_j - vx_i;
                float relativeVel_y = vy_j - vy_i;
                float relativeVel_z = vz_j - vz_i;
                float relativeSpeed = sqrtf(relativeVel_x * relativeVel_x
                    + relativeVel_y * relativeVel_y
                    + relativeVel_z * relativeVel_z);


                float prob = d_conf.hss_collisionProbMultiplier * N_x * sqrtf(relativeSpeed);
                if (curand_uniform(&rngState) < prob) {
                    //                    4. collide
                    float N[3];
                    random_isotropic_vector_device(N, &rngState);
                    relativeSpeed *= 0.5f;
                    // float VC[3] = {0.5f * (vx_j + vx_i), 0.5f * (vy_j + vy_i), 0.5f * (vz_j + vz_i)};
                    float VC_x = 0.5f * (vx_j + vx_i);
                    float VC_y = 0.5f * (vy_j + vy_i);
                    float VC_z = 0.5f * (vz_j + vz_i);
                    // float VCr[3] = {relativeSpeed * N[0], relativeSpeed * N[1], relativeSpeed * N[2]};
                    float VCr_x = relativeSpeed * N[0];
                    float VCr_y = relativeSpeed * N[1];
                    float VCr_z = relativeSpeed * N[2];
                    P.vx[i_global] = VC_x + VCr_x;
                    P.vy[i_global] = VC_y + VCr_y;
                    P.vz[i_global] = VC_z + VCr_z;
                    P.vx[j_global] = VC_x - VCr_x;
                    P.vy[j_global] = VC_y - VCr_y;
                    P.vz[j_global] = VC_z - VCr_z;

                    collisions++;
                }
            }
        }
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
        if (NPC > d_conf.hss_threshold) {
            // call child kernel with stream
            cudaStream_t stream;
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            child_hss_scheme_kernel<<<1, 64, 0, stream>>>(total_collisions, P, NPC, (NPC % 2 == 0) ? NPC - 1 : NPC, cellList, rngStates, idx);
            cudaStreamDestroy(stream);
            // no need to sync because of atomic add to collisions, only sync at very end
        }
        else if (NPC >= 2) {

            int *IPC = &cellList[IDX_LIST(k, l, m, 0)];

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
                int i = IPC[i_local];
                int j = IPC[j_local];

                float vx_i = P.vx[i];
                float vy_i = P.vy[i];
                float vz_i = P.vz[i];

                float vx_j = P.vx[j];
                float vy_j = P.vy[j];
                float vz_j = P.vz[j];

                float relativeVel[3] = { vx_j - vx_i, vy_j - vy_i, vz_j - vz_i };
                float relativeSpeed = sqrt(relativeVel[0] * relativeVel[0]
                                            + relativeVel[1] * relativeVel[1]
                                            + relativeVel[2] * relativeVel[2]);

                // The real value to be calculated:
                // double collisionProb = d_conf.ntcs_collisionProbMultiplier * pow(1.0 / relativeSpeed, d_conf.ntcs_collisionProbExponent) * relativeSpeed;
                // But, we assume omega=0.75, which lets us remove pow() in a simple way:
                float collisionProb = d_conf.ntcs_collisionProbMultiplier * sqrt(relativeSpeed);
                if (curand_uniform(&rngState) < collisionProb) {
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
