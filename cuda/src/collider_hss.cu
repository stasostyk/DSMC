#include "../include/collider_hss.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/particles.h"
#include "../include/simulation.h"
#include "../include/cuda_utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

__global__ void hss_scheme_kernel(unsigned long long *total_collisions,
                                  Particles P,
                                  int *cellCount,
                                  int *cellList,
                                  curandState *rngStates) {
    __shared__ int collisionsBlock[4];
    __shared__ int localParticleList[MAX_PARTICLES_PER_CELL];
    __shared__ int NPC;
    __shared__ int N_x;

    int blockSize = blockDim.x;

    int tid = threadIdx.x;

    collisionsBlock[tid] = 0;

    if (blockIdx.x < NX && blockIdx.y < NY && blockIdx.z < NZ) {
        int cell_idx = IDX_CELL(blockIdx.x, blockIdx.y, blockIdx.z);

        if (threadIdx.x == 0) {
            NPC = cellCount[cell_idx];
            N_x = (NPC % 2 == 0) ? NPC - 1 : NPC;
        }
        __syncthreads();

        if (NPC < d_conf.hss_threshold) return;

        int rngIdx = IDX_CELL(blockIdx.x, blockIdx.y, blockIdx.z) * blockSize + tid;
        curandState rngState = rngStates[rngIdx];

        int nPairs = NPC / 2;
        int offset = (NPC + 1) / 2;
        for (int i = tid; i < NPC; i += blockSize) {
            localParticleList[i] =
                    cellList[IDX_LIST(blockIdx.x, blockIdx.y, blockIdx.z, i)];
        }
        __syncthreads();

        for (int b = 0; b < d_conf.hss_nbatch; b++) {
            // // fisher-yates shuffle (could be optimized with parallel sorting of random keys)
            // if (threadIdx.x == 0) {
            //     for (int i = 0; i < NPC; i++) {
            //         int j = i + (int) (curand_uniform(&rngState) * (NPC - i));
            //         if (j < NPC) {
            //             int temp = localParticleList[i];
            //             localParticleList[i] = localParticleList[j];
            //             localParticleList[j] = temp;
            //         }
            //     }
            // }
            // __syncthreads();

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

                    float relativeVel[3] = {vx_j - vx_i, vy_j - vy_i, vz_j - vz_i};
                    float relativeSpeed = sqrt(relativeVel[0] * relativeVel[0]
                                                + relativeVel[1] * relativeVel[1]
                                                + relativeVel[2] * relativeVel[2]);


                    float prob = d_conf.hss_collisionProbMultiplier * N_x * relativeSpeed;
                    if (curand_uniform(&rngState) < prob) {
                        //                    4. collide
                        float N[3];
                        random_isotropic_vector_device(N, &rngState);
                        relativeSpeed *= 0.5f;
                        float VC[3] = {0.5f * (vx_j + vx_i), 0.5f * (vy_j + vy_i), 0.5f * (vz_j + vz_i)};
                        float VCr[3] = {relativeSpeed * N[0], relativeSpeed * N[1], relativeSpeed * N[2]};
                        P.vx[i_global] = VC[0] + VCr[0];
                        P.vy[i_global] = VC[1] + VCr[1];
                        P.vz[i_global] = VC[2] + VCr[2];
                        P.vx[j_global] = VC[0] - VCr[0];
                        P.vy[j_global] = VC[1] - VCr[1];
                        P.vz[j_global] = VC[2] - VCr[2];

                        collisionsBlock[tid] += 1;
                    }
                }

            }
            __syncthreads();
        }
        rngStates[rngIdx] = rngState;
    }

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

void collide_particles_hss(Simulation *sim) {
    dim3 threadsPerBlock(4);
    dim3 blocksPerGrid(NX, NY, NZ);

    hss_scheme_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            sim->d_totalCollisions, sim->d_P, sim->d_cellCount, sim->d_cellList, sim->rngStates
    );
    CHECK_KERNELCALL();

}