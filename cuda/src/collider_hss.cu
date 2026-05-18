#include "../include/collider_hss.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/particles.h"
#include "../include/simulation.h"
#include "../include/cuda_utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


__global__ void hss_scheme_kernel(unsigned long long *total_collisions,
                                  Particles P,
                                  int *cellCount,
                                  int *cellList,
                                  curandState *rngStates) {
    __shared__ int collisionsBlock[64];

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z * blockDim.z + threadIdx.z;

    int collisions = 0;

    if (k < NX && l < NY && m < NZ) {
        int idx = IDX_CELL(k, l, m);

//    1. generate local particle index list (we can use cellList)
//    2. pair the particles first half and second half
//    3. collide all pairs

        curandState rngState = rngStates[idx];
        int NPC = cellCount[idx];
        // create a local list for shuffling
        int localParticleList[MAX_PARTICLES_PER_CELL];
        for (int i = 0; i < NPC; i++) {
            localParticleList[i] = cellList[IDX_LIST(k, l, m, i)];
        }

        int N_x = (NPC % 2 == 0) ? NPC - 1 : NPC;
        if (NPC >= 2) {
            for (int b = 0; b < d_conf.hss_nbatch; b++) {
//                5. fisher-yates shuffle
                for (int i = 0; i < NPC; i++) {
                    int j = i + (int) (curand_uniform(&rngState) * (NPC - i));
                    if (j < NPC) {
                        int temp = localParticleList[i];
                        localParticleList[i] = localParticleList[j];
                        localParticleList[j] = temp;
                    }
                }

                for (int i = 0; i < NPC / 2; i++) {
                    int j = i + (NPC + 1) / 2;
                    if (j < NPC) {
                        int i_global = localParticleList[i];
                        int j_global = localParticleList[j];

                        double vx_i = P.vx[i_global];
                        double vy_i = P.vy[i_global];
                        double vz_i = P.vz[i_global];

                        double vx_j = P.vx[j_global];
                        double vy_j = P.vy[j_global];
                        double vz_j = P.vz[j_global];

                        double relativeVel[3] = {vx_j - vx_i, vy_j - vy_i, vz_j - vz_i};
                        double relativeSpeed = sqrt(relativeVel[0] * relativeVel[0]
                                                    + relativeVel[1] * relativeVel[1]
                                                    + relativeVel[2] * relativeVel[2]);


                        double prob = d_conf.hss_collisionProbMultiplier * N_x * relativeSpeed;
                        if (curand_uniform(&rngState) < prob) {
                            //                    4. collide
                            double N[3];
                            random_isotropic_vector_device(N, &rngState);
                            relativeSpeed *= 0.5;
                            double VC[3] = {0.5 * (vx_j + vx_i), 0.5 * (vy_j + vy_i), 0.5 * (vz_j + vz_i)};
                            double VCr[3] = {relativeSpeed * N[0], relativeSpeed * N[1], relativeSpeed * N[2]};
                            P.vx[i_global] = VC[0] + VCr[0];
                            P.vy[i_global] = VC[1] + VCr[1];
                            P.vz[i_global] = VC[2] + VCr[2];
                            P.vx[j_global] = VC[0] - VCr[0];
                            P.vy[j_global] = VC[1] - VCr[1];
                            P.vz[j_global] = VC[2] - VCr[2];

                            collisions++;
                        }
                    }
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

void collide_particles_hss(Simulation *sim) {
    dim3 threadsPerBlock(4, 4, 4);
    dim3 blocksPerGrid(
            (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    hss_scheme_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            sim->d_totalCollisions, sim->d_P, sim->d_cellCount, sim->d_cellList, sim->rngStates
    );
    CHECK_KERNELCALL();

}