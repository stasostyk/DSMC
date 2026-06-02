#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../include/simulation.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/cuda_utils.h"
#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

void swap_particles_with_new(Simulation *sim) {
    // swap d_P and d_new_P
    Particles temp = sim->d_P;
    sim->d_P = sim->d_new_P;
    sim->d_new_P = temp;
}

__global__ void init_rng_kernel(curandState *states, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAX_PARTICLES) return;
    curand_init(seed, i, 0, &states[i]);
}

void setup(Simulation *sim, Config *conf) {
    sim->conf = (Config *) malloc(sizeof(Config));

    memcpy(sim->conf, conf, sizeof(Config));
    CHECK(cudaMemcpyToSymbol(d_conf, conf, sizeof(Config)));

    CHECK(cudaMalloc(&(sim->d_new_NP), sizeof(int)));

    // initialize the SoA
    sim->P.x = (float *)malloc(PARTICLES_FIELD_SZ);
    sim->P.y = (float *)malloc(PARTICLES_FIELD_SZ);
    sim->P.z = (float *)malloc(PARTICLES_FIELD_SZ);
    sim->P.vx = (float *)malloc(PARTICLES_FIELD_SZ);
    sim->P.vy = (float *)malloc(PARTICLES_FIELD_SZ);
    sim->P.vz = (float *)malloc(PARTICLES_FIELD_SZ);

    sim->samples = (Cell *)malloc(SAMPLES_SZ);

    CHECK(cudaMalloc(&sim->rngStates, MAX_PARTICLES * sizeof(curandState)));
//  device SoA for particles
    CHECK(cudaMalloc(&sim->d_P.x,  PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_P.y,  PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_P.z,  PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_P.vx, PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_P.vy, PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_P.vz, PARTICLES_FIELD_SZ));
//  device SoA for new particles
    CHECK(cudaMalloc(&sim->d_new_P.x,  PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_new_P.y,  PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_new_P.z,  PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_new_P.vx, PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_new_P.vy, PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_new_P.vz, PARTICLES_FIELD_SZ));

    CHECK(cudaMalloc(&sim->d_samples, SAMPLES_SZ));
    CHECK(cudaMalloc(&sim->d_cellCount, CELL_COUNT_SZ));
    CHECK(cudaMalloc(&sim->d_cellCountPrefixSum, CELL_COUNT_SZ));

    CHECK(cudaMalloc(&sim->d_valid, sizeof(int) * MAX_PARTICLES));
    CHECK(cudaMalloc(&sim->d_particleIds, sizeof(int) * MAX_PARTICLES));
    CHECK(cudaMalloc(&sim->d_particleIdsSorted, sizeof(int) * MAX_PARTICLES));
    CHECK(cudaMalloc(&sim->d_cellKeys, sizeof(int) * MAX_PARTICLES));
    CHECK(cudaMalloc(&sim->d_cellKeysSorted, max(CELL_COUNT_SZ, sizeof(int) * MAX_PARTICLES)));

    sim->temp_storage_bytes = 0;
    sim->d_temp_storage = nullptr;

    cub::DeviceScan::ExclusiveSum(
        sim->d_temp_storage, 
        sim->temp_storage_bytes, 
        sim->d_cellCount, 
        sim->d_cellCountPrefixSum, 
        NX*NY*NZ);
    cudaMalloc(&sim->d_temp_storage, sim->temp_storage_bytes);


    int bits = 0;
    int ncells = NX * NY * NZ;
    while ((1<<bits) < ncells) bits++;
    sim->radixSortBits = bits;
    // Also check space for radix sort
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, sort_temp_bytes,
        sim->d_cellKeys, sim->d_cellKeysSorted,
        sim->d_particleIds, sim->d_particleIdsSorted,  // repurpose as value buffer
        MAX_PARTICLES,
        0, sim->radixSortBits
    );
    // Reallocate if needed (lazy resize)
    if (sort_temp_bytes > sim->temp_storage_bytes) {
        cudaFree(sim->d_temp_storage);
        cudaMalloc(&sim->d_temp_storage, sort_temp_bytes);
        sim->temp_storage_bytes = sort_temp_bytes;
    }

    // Also check for exclusive sum again
    size_t new_temp_bytes;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        new_temp_bytes,
        sim->d_valid,
        sim->d_particleIds,
        MAX_PARTICLES
    );
    // Reallocate if needed (lazy resize)
    if (new_temp_bytes > sim->temp_storage_bytes) {
        cudaFree(sim->d_temp_storage);
        cudaMalloc(&sim->d_temp_storage, new_temp_bytes);
        sim->temp_storage_bytes = new_temp_bytes;
    }


    sim->sampleSteps = 0;
    sim->NP = 0;
    sim->totalCollisions = 0;

    cudaMalloc(&sim->d_totalCollisions, sizeof(unsigned long long));
    cudaMemset(sim->d_totalCollisions, 0, sizeof(unsigned long long));
    cudaMemset(sim->d_samples, 0, SAMPLES_SZ);

    sim->NP = NX * NY * NZ * conf->particlesPerCellTarget;

    if (sim->NP > MAX_PARTICLES) {
        fprintf(stderr, "Too many particles for MAX_PARTICLES\n");
        exit(1);
    }

    // init randomness
    int threads = 256;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((MAX_PARTICLES + threads - 1) / threads, 1, 1);

    unsigned long seed = (unsigned long)time(NULL);
    init_rng_kernel<<<blocksPerGrid, threadsPerBlock>>>(sim->rngStates, seed);
    CHECK_KERNELCALL();
}

// __global__ void reorder_particles_by_cell_kernel(
//     Particles P_out, Particles P, int *cellCount, int *cellList, int *cellCountPrefixSum, int NP
// ) {

//     int cell_idx = IDX_CELL(blockIdx.x, blockIdx.y, blockIdx.z);
//     int NPC = cellCount[cell_idx];
//     for (int i = threadIdx.x; i < NPC; i += blockDim.x) {
//         int cid = cell_idx * MAX_PARTICLES_PER_CELL + i;
//         int pid = cellList[cid];
//         int offset = cellCountPrefixSum[cell_idx] + i;
//         P_out.x[offset] = P.x[pid];
//         P_out.y[offset] = P.y[pid];
//         P_out.z[offset] = P.z[pid];
//         P_out.vx[offset] = P.vx[pid];
//         P_out.vy[offset] = P.vy[pid];
//         P_out.vz[offset] = P.vz[pid];
    
//         cellList[cid] = offset;
//     }
// }


// void reorder_particles_by_cell(Simulation *sim) {

//     // Run exclusive prefix sum
//     cub::DeviceScan::ExclusiveSum(
//         sim->d_temp_storage, 
//         sim->temp_storage_bytes,
//         sim->d_cellCount, 
//         sim->d_cellCountPrefixSum, 
//         NX*NY*NZ
//     );

//     dim3 blocksPerGrid(NX, NY, NZ);

//     reorder_particles_by_cell_kernel<<<blocksPerGrid, 32>>>(
//         sim->d_new_P, sim->d_P, sim->d_cellCount, sim->d_cellList, sim->d_cellCountPrefixSum, sim->NP
//     );
//     CHECK_KERNELCALL();

//     // swap d_P and d_new_P
//     Particles temp = sim->d_P;
//     sim->d_P = sim->d_new_P;
//     sim->d_new_P = temp;
// }

// __global__ void bin_particles_kernel(
//     Particles P, int *cellCount, int *cellList, int *cellKeys, int NP
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= NP) return;

//     int k = __float2int_rd(P.x[i] * d_conf.inv_dx);
//     int l = __float2int_rd(P.y[i] * d_conf.inv_dy);
//     int m = __float2int_rd(P.z[i] * d_conf.inv_dz);

//     int j = cellKeys[i];

//     int n = atomicAdd(&cellCount[j], 1);
//     cellList[IDX_LIST(k, l, m, n)] = i;
// }

// void index_particles(Simulation *sim) {
//     cudaMemset(sim->d_cellCount, 0, CELL_COUNT_SZ);

//     int thr = 128;
//     dim3 threadsPerBlock2(thr, 1, 1);
//     dim3 blocksPerGrid2((sim->NP + thr - 1) / thr, 1, 1);
//     bin_particles_kernel<<<blocksPerGrid2, threadsPerBlock2>>>(
//         sim->d_P, sim->d_cellCount, sim->d_cellList, sim->d_cellKeys, sim->NP
//     );
//     CHECK_KERNELCALL();
// }

__global__ void generate_particles_in_rect_kernel(
    Particles P,
    int start,
    int Nnew,
    float x1, float x2,
    float y1, float y2,
    float z1, float z2,
    int moveFlag,
    curandState *rngStates
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nnew) return;

    float ux = d_conf.UxFree;
    float uy = d_conf.UyFree;
    float uz = d_conf.UzFree;
    float dt = d_conf.dt;

    int idx = start + i;

    curandState rngState = rngStates[idx];

    float rx = curand_uniform(&rngState);
    float ry = curand_uniform(&rngState);
    float rz = curand_uniform(&rngState);

    // TODO possibly better to create everything in local variable, and only then store in P?
    P.x[idx] = x1 + (x2 - x1) * rx;
    P.y[idx] = y1 + (y2 - y1) * ry;
    P.z[idx] = z1 + (z2 - z1) * rz;

    float vx = curand_normal(&rngState) * d_conf.generation_derivatedMultiplier + ux;
    float vy = curand_normal(&rngState) * d_conf.generation_derivatedMultiplier + uy;
    float vz = curand_normal(&rngState) * d_conf.generation_derivatedMultiplier + uz;

    P.vx[idx] = vx;
    P.vy[idx] = vy;
    P.vz[idx] = vz;

    if (moveFlag) {
        P.x[idx] += dt * vx;
        P.y[idx] += dt * vy;
        P.z[idx] += dt * vz;
    }

    rngStates[idx] = rngState;
}


void generate_particles_in_rect(
    Simulation *sim,
    float x1, float x2,
    float y1, float y2,
    float z1, float z2,
    int moveFlag
) {
    double ngas = sim->conf->NFree;

    double V = (x2 - x1) * (y2 - y1) * (z2 - z1);
    double N_add_exp = ngas * V / sim->conf->weight;
    int Nnew;
    if ( N_add_exp > 20.0 ) { // use uniform approximation if large, poisson if small
        Nnew = (int)(N_add_exp);
        if (randu() < N_add_exp - Nnew) Nnew++;
    } else {
        Nnew = (int) randp( N_add_exp );
    }

    if (sim->NP + Nnew > MAX_PARTICLES) {
        fprintf(stderr, "Too many particles\n");
        exit(1);
    }

    int threads = 128;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((Nnew + threads - 1) / threads, 1, 1);

    generate_particles_in_rect_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_P, sim->NP, Nnew, 
        x1, x2, y1, y2, z1, z2, 
        moveFlag, sim->rngStates
    );
    CHECK_KERNELCALL();

    sim->NP += Nnew;
}

__global__ void mark_valid_kernel(Particles P, int *valid, int NP) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    valid[i] =
        (P.x[i] >= 0.0f && P.x[i] < d_conf.Lx &&
         P.y[i] >= 0.0f && P.y[i] < d_conf.Ly &&
         P.z[i] >= 0.0f && P.z[i] < d_conf.Lz);
}

// Removes the particles that were created inside the balls.
// This is called only once, at the initialization of the 
// particles inside the whole volume, which happens before
// the main simulation loop.
__global__ void filter_particles_inside_ball(
    Particles P, Particles P_out, int NP, int *new_NP
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    bool insideBall = false;

    for (int ballId = 0; ballId < d_conf.ballCnt; ballId++) {
        float cx = d_conf.balls[ballId].ballCenterX;
        float cy = d_conf.balls[ballId].ballCenterY;
        float cz = d_conf.balls[ballId].ballCenterZ;
        float R2 = d_conf.balls[ballId].ballRadiusSquared;

        float rx = P.x[i] - cx;
        float ry = P.y[i] - cy;
        float rz = P.z[i] - cz;

        insideBall |= (rx*rx + ry*ry + rz*rz <= R2);
    }

    if (!insideBall) {
        int pos = atomicAdd(new_NP, 1);
        P_out.x[pos] = P.x[i];
        P_out.y[pos] = P.y[i];
        P_out.z[pos] = P.z[i];
        P_out.vx[pos] = P.vx[i];
        P_out.vy[pos] = P.vy[i];
        P_out.vz[pos] = P.vz[i];
    }
}

void remove_particles_inside_balls(Simulation *sim) {
    int threads = 128;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((sim->NP + threads - 1) / threads, 1, 1);

    CHECK(cudaMemset(sim->d_new_NP, 0, sizeof(int)));

    filter_particles_inside_ball<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_P, sim->d_new_P, sim->NP, sim->d_new_NP
    );
    CHECK_KERNELCALL();

    int host_new_NP;
    CHECK(cudaMemcpy(&host_new_NP, sim->d_new_NP, sizeof(int), cudaMemcpyDeviceToHost));
    sim->NP = host_new_NP;
    
    swap_particles_with_new(sim);
}

void initialize_particles(Simulation *sim) {
    sim->NP = 0;
    generate_particles_in_rect(sim, 0.0, sim->conf->Lx, 0.0, sim->conf->Ly, 0.0, sim->conf->Lz, 0);

    remove_particles_inside_balls(sim);
}

// __global__ void filter_particles_out_of_bounds(
//     Particles P, Particles P_out, int NP, int *new_NP
// ) {
//     // TODO this can be possibly optimized:
//     // 1st kernel: mark the particles true/false if they need to be removed
//     // 2nd kernel: scan/prefix sum 
//     // 3rd kernel: (scatter) collect valid particles

//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= NP) return;

//     // check if particle valid
//     if (P.x[i] >= 0.0 && P.x[i] < d_conf.Lx &&
//         P.y[i] >= 0.0 && P.y[i] < d_conf.Ly &&
//         P.z[i] >= 0.0 && P.z[i] < d_conf.Lz
//     ) {
//         int pos = atomicAdd(new_NP, 1);
//         P_out.x[pos] = P.x[i];
//         P_out.y[pos] = P.y[i];
//         P_out.z[pos] = P.z[i];
//         P_out.vx[pos] = P.vx[i];
//         P_out.vy[pos] = P.vy[i];
//         P_out.vz[pos] = P.vz[i];
//     }
// }

void apply_boundary_conditions_free_stream(Simulation *sim) {
    Config *conf = sim->conf;
    generate_particles_in_rect(sim, -(conf->DL), 0.0, 0.0, conf->Ly, 0.0, conf->Lz, 1);
    generate_particles_in_rect(sim, conf->Lx, conf->Lx + conf->DL, 0.0, conf->Ly, 0.0, conf->Lz, 1);
    generate_particles_in_rect(sim, 0.0, conf->Lx, -(conf->DL), 0.0, 0.0, conf->Lz, 1);
    generate_particles_in_rect(sim, 0.0, conf->Lx, conf->Ly, conf->Ly + conf->DL, 0.0, conf->Lz, 1);
    generate_particles_in_rect(sim, 0.0, conf->Lx, 0.0, conf->Ly, -(conf->DL), 0.0, 1);
    generate_particles_in_rect(sim, 0.0, conf->Lx, 0.0, conf->Ly, conf->Lz, conf->Lz + conf->DL, 1);
}

__global__ void scatter_and_key_kernel(
    Particles P, Particles P_out, int *valid, int *particleIds, 
    int *cellKeys, int *cellCount,
    int NP
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    if (!valid[i]) return;

    int j = particleIds[i];

    float x = P.x[i];
    float y = P.y[i];
    float z = P.z[i];

    // write compact particle
    P_out.x[j]  = x;
    P_out.y[j]  = y;
    P_out.z[j]  = z;
    P_out.vx[j] = P.vx[i];
    P_out.vy[j] = P.vy[i];
    P_out.vz[j] = P.vz[i];

    // compute cell key
    int k = __float2int_rd(x * d_conf.inv_dx);
    int l = __float2int_rd(y * d_conf.inv_dy);
    int m = __float2int_rd(z * d_conf.inv_dz);

    int cell = IDX_CELL(k,l,m);
    cellKeys[j] = cell;
    atomicAdd(&cellCount[cell], 1);
}

// __global__ void gather_kernel(
//     Particles P_in, Particles P_out,
//     int *sortedIds,          // output of SortPairs: particle indices in cell order
//     int *sortedCellKeys, int *cellCount,
//     int NP
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= NP) return;

//     int src = sortedIds[i];   // which compacted particle goes to slot i

//     P_out.x[i]  = P_in.x[src];
//     P_out.y[i]  = P_in.y[src];
//     P_out.z[i]  = P_in.z[src];
//     P_out.vx[i] = P_in.vx[src];
//     P_out.vy[i] = P_in.vy[src];
//     P_out.vz[i] = P_in.vz[src];

//     atomicAdd(&cellCount[sortedCellKeys[i]], 1);
// }

// __global__ void rebuild_cell_count_kernel(
//     int *sortedCellKeys, int *cellCount, int NP
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= NP) return;
//     atomicAdd(&cellCount[sortedCellKeys[i]], 1);
// }

// Already have cellCount from scatter_and_key_kernel (you compute cellKeys[j] there)
// Step 1: prefix sum over cellCount → gives start offset per cell  
// Step 2: scatter particles to final position using atomicAdd per cell

__global__ void counting_sort_scatter_kernel(
    Particles P_in, Particles P_out,
    int *cellKeys,           // cell of each compacted particle
    int *cellOffsets,        // prefix sum of cellCount (write cursor per cell)
    // int *cellCount,
    int NP
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    int cell = cellKeys[i];
    int dest = atomicAdd(&cellOffsets[cell], 1);  // claim a slot

    P_out.x[dest]  = P_in.x[i];
    P_out.y[dest]  = P_in.y[i];
    P_out.z[dest]  = P_in.z[i];
    P_out.vx[dest] = P_in.vx[i];
    P_out.vy[dest] = P_in.vy[i];
    P_out.vz[dest] = P_in.vz[i];

}

void filter_and_index_particles(Simulation *sim) {
    int threads = 128;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((sim->NP + threads - 1) / threads, 1, 1);

    mark_valid_kernel<<<blocksPerGrid, threadsPerBlock>>>(sim->d_P, sim->d_valid, sim->NP);
    CHECK_KERNELCALL();



    // valid -> particle new index map
    cub::DeviceScan::ExclusiveSum(
        sim->d_temp_storage,
        sim->temp_storage_bytes,
        sim->d_valid,
        sim->d_particleIds,
        sim->NP
    );

    int h_lastPrefixVal, h_lastValid;
    CHECK(cudaMemcpy(&h_lastPrefixVal, sim->d_particleIds + sim->NP - 1,
               sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_lastValid, sim->d_valid + sim->NP - 1,
               sizeof(int), cudaMemcpyDeviceToHost));
    int h_newNP = h_lastPrefixVal + h_lastValid;
   
    cudaMemset(sim->d_cellCount, 0, CELL_COUNT_SZ);

    scatter_and_key_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_P, sim->d_new_P,
        sim->d_valid, sim->d_particleIds,
        sim->d_cellKeys, 
        sim->d_cellCount,
        sim->NP
    );
    CHECK_KERNELCALL();

    // DO SORT


    dim3 blocksNew((h_newNP + threads - 1) / threads, 1, 1);
    // rebuild_cell_count_kernel<<<blocksNew, threadsPerBlock>>>(
    //     sim->d_cellKeys, sim->d_cellCount, h_newNP
    // );
    // CHECK_KERNELCALL();

    cub::DeviceScan::ExclusiveSum(
        sim->d_temp_storage, sim->temp_storage_bytes,
        sim->d_cellCount, sim->d_cellCountPrefixSum,
        NX * NY * NZ
    );

    // Copy prefix sum into a mutable "cursor" array (reuse d_particleIds as scratch)
    CHECK(cudaMemcpy(sim->d_cellKeysSorted, sim->d_cellCountPrefixSum,
            sizeof(int) * NX * NY * NZ, cudaMemcpyDeviceToDevice));

    // CHECK(cudaMemset(sim->d_cellCount, 0, CELL_COUNT_SZ));
    counting_sort_scatter_kernel<<<blocksNew, threadsPerBlock>>>(
        sim->d_new_P, sim->d_P,
        sim->d_cellKeys,
        sim->d_cellKeysSorted,   // cursor (gets incremented)
        // sim->d_cellCount,
        h_newNP
    );
    CHECK_KERNELCALL();




    // // Values to sort: we want particle positions (0..h_newNP-1) 
    // // so we can gather them in cell order.
    // // Re-initialize d_particleIds as 0,1,2,...,h_newNP-1
    // thrust::sequence(
    //     thrust::device,
    //     sim->d_particleIds,
    //     sim->d_particleIds + h_newNP
    // );

    // cub::DeviceRadixSort::SortPairs(
    //     sim->d_temp_storage,
    //     sim->temp_storage_bytes,
    //     sim->d_cellKeys,          // keys in  (cell index of compacted particle i)
    //     sim->d_cellKeysSorted,    // keys out
    //     sim->d_particleIds,       // values in  (0,1,2,...,h_newNP-1)
    //     sim->d_particleIdsSorted, // values out (permutation)
    //     h_newNP,
    //     0, sim->radixSortBits
    // );

    // cudaMemset(sim->d_cellCount, 0, CELL_COUNT_SZ);

    // dim3 blocksNew((h_newNP + threads - 1) / threads, 1, 1);
    // gather_kernel<<<blocksNew, threadsPerBlock>>>(
    //     sim->d_new_P, sim->d_P,          // src=compacted, dst=final
    //     sim->d_particleIdsSorted,
    //     sim->d_cellKeysSorted, sim->d_cellCount,
    //     h_newNP
    // );
    // CHECK_KERNELCALL();

    sim->NP = h_newNP;

    // // THEN COUNT PREF SUMS
    // cub::DeviceScan::ExclusiveSum(
    //     sim->d_temp_storage,
    //     sim->temp_storage_bytes,
    //     sim->d_cellCount,
    //     sim->d_cellCountPrefixSum,
    //     NX * NY * NZ
    // );

    // CHECK(cudaMemset(sim->d_new_NP, 0, sizeof(int)));

    // filter_particles_out_of_bounds<<<blocksPerGrid, threadsPerBlock>>>(
    //     sim->d_P, sim->d_new_P, sim->NP, sim->d_new_NP
    // );
    // CHECK_KERNELCALL();

    // int host_new_NP;
    // CHECK(cudaMemcpy(&host_new_NP, sim->d_new_NP, sizeof(int), cudaMemcpyDeviceToHost));
    // sim->NP = host_new_NP;
    
    // swap_particles_with_new(sim);
}

__global__ void move_particles_kernel(Particles P, int NP, curandState *rngStates) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    // TODO a lot of calls to global memory P[i], probably would be better to save to local

    float X0 = P.x[i];
    float Y0 = P.y[i];
    float Z0 = P.z[i];
    P.x[i] += d_conf.dt * P.vx[i];
    P.y[i] += d_conf.dt * P.vy[i];
    P.z[i] += d_conf.dt * P.vz[i];

    curandState rngState = rngStates[i];

    // CHECK WINGS
    for (int wingId = 0; wingId < d_conf.wingCnt; wingId++) {
        float dt = d_conf.dt;
        double moleculeMass = d_conf.moleculeMass;
        double Tw = d_conf.wings[wingId].Tw;
        float WingX = d_conf.wings[wingId].WingX;
        float WingY = d_conf.wings[wingId].WingY;
        float WingLength = d_conf.wings[wingId].WingLength;

        if ( ( Y0 - WingY ) * ( P.y[i] - WingY ) < 0.0 ) {
            // Linear interpolation to point Y = WingY
            float Xw=( X0*(WingY-P.y[i])+P.x[i]*(Y0-WingY))/(Y0-P.y[i]);
            float Zw=( Z0*(WingY-P.y[i])+P.z[i]*(Y0-WingY))/(Y0-P.y[i]);
            if ( Zw < 0.3 || Zw > 0.7 ) return; // wing only occupies 0.3 < z < 0.7
            if ( Xw > WingX && Xw < WingX + WingLength ) {
                // Molecule interacts with the wing during the time step
                // Linear interpolation of the time of scattering, Eq. (6.5.4)
                float Dt1 = dt - dt * ( Y0 - WingY ) / ( Y0 - P.y[i] );
                // Generate velocity vector of the reflected molecule
                diffuse_scattering_y_device(
                    &(P.vx[i]), &(P.vy[i]), &(P.vz[i]),
                    moleculeMass,Tw,(Y0-WingY>0)?1.0:(-1.0),
                    d_conf.KB,
                    &rngState
                );
                // Move the reflected molecule
                P.x[i] = Xw + Dt1 * P.vx[i];
                P.y[i] = WingY + Dt1 * P.vy[i];
            }
        }
    }


    // CHECK BALLS
    for (int ballId = 0; ballId < d_conf.ballCnt; ballId++) {
        // Ray-sphere intersection test
        
        // Initial and final positions
        float x0 = X0, y0 = Y0, z0 = Z0;
        float x1 = P.x[i], y1 = P.y[i], z1 = P.z[i];

        // Direction of motion
        float dx = x1 - x0;
        float dy = y1 - y0;
        float dz = z1 - z0;

        // Sphere center
        float cx = d_conf.balls[ballId].ballCenterX;
        float cy = d_conf.balls[ballId].ballCenterY;
        float cz = d_conf.balls[ballId].ballCenterZ;
        float ballRadius = d_conf.balls[ballId].ballRadius;

        // Shifted initial position
        float rx = x0 - cx;
        float ry = y0 - cy;
        float rz = z0 - cz;

        // Quadratic coefficients: |r + t d|^2 = R^2
        float a = dx*dx + dy*dy + dz*dz;
        float b = 2.0 * (rx*dx + ry*dy + rz*dz);
        float c = rx*rx + ry*ry + rz*rz - ballRadius*ballRadius;

        float disc = b*b - 4.0*a*c;

        if (disc >= 0.0) {
            float sqrt_disc = sqrt(disc);

            // time solutions
            float t1 = (-b - sqrt_disc) / (2.0*a);
            float t2 = (-b + sqrt_disc) / (2.0*a);

            // pick earliest valid intersection in [0,1]
            float t_hit = -1.0;
            if (t1 >= 0.0 && t1 <= 1.0) t_hit = t1;
            else if (t2 >= 0.0 && t2 <= 1.0) t_hit = t2;

            if (t_hit >= 0.0) {
                // Intersection point
                float Xw = x0 + t_hit * dx;
                float Yw = y0 + t_hit * dy;
                float Zw = z0 + t_hit * dz;

                // Remaining time after collision
                float Dt1 = d_conf.dt * (1.0 - t_hit);

                // Surface normal (outward)
                float nx = (Xw - cx) / ballRadius;
                float ny = (Yw - cy) / ballRadius;
                float nz = (Zw - cz) / ballRadius;

                // Diffuse reflection aligned with normal
                diffuse_scattering_device(&(P.vx[i]), &(P.vy[i]), &(P.vz[i]),
                                d_conf.moleculeMass, d_conf.balls[ballId].Tb,
                                nx, ny, nz, d_conf.KB,
                                &rngState
                            );

                // Move after collision
                P.x[i] = Xw + Dt1 * P.vx[i];
                P.y[i] = Yw + Dt1 * P.vy[i];
                P.z[i] = Zw + Dt1 * P.vz[i];
            }
        }
    }

    rngStates[i] = rngState;
}

void move_particles(Simulation *sim) {
    int threads = 128;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((sim->NP + threads - 1) / threads, 1, 1);

    move_particles_kernel<<<blocksPerGrid, threadsPerBlock>>>(sim->d_P, sim->NP, sim->rngStates);
    CHECK_KERNELCALL();
}

__global__ void accumulate_sampling_kernel(
    int *cellCount, int *cellCountPrefixSum, Cell *samples, Particles P
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z * blockDim.z + threadIdx.z;

    if (k >= NX || l >= NY || m >= NZ) return;
    
    int cellIdx = IDX_CELL(k, l, m);
    int offset = cellCountPrefixSum[cellIdx];

    int Nc = cellCount[cellIdx];
    samples[cellIdx].countNP += Nc;

    for (int q = 0; q < Nc; q++) {
        // int i = cellList[IDX_LIST(k, l, m, q)];
        int i = offset + q;

        float vx = P.vx[i];
        float vy = P.vy[i];
        float vz = P.vz[i];

        samples[IDX_CELL(k, l, m)].countVx += vx;
        samples[IDX_CELL(k, l, m)].countVy += vy;
        samples[IDX_CELL(k, l, m)].countVz += vz;
        samples[IDX_CELL(k, l, m)].countV2 += vx * vx + vy * vy + vz * vz;
    }
}


void accumulate_sampling(Simulation *sim) {
    sim->sampleSteps++;

    dim3 threadsPerBlock(4, 4, 4);
    dim3 blocksPerGrid(
        (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    accumulate_sampling_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_cellCount, sim->d_cellCountPrefixSum, sim->d_samples, sim->d_P
    );
    CHECK_KERNELCALL();
}

void clearPointers(Simulation *sim) {
    free(sim->conf);
//    free(sim->P);
    free(sim->P.x);
    free(sim->P.y);
    free(sim->P.z);
    free(sim->P.vx);
    free(sim->P.vy);
    free(sim->P.vz);
    free(sim->samples);

    CHECK(cudaFree(sim->rngStates));

//    CHECK(cudaFree(sim->d_P));
    CHECK(cudaFree(sim->d_P.x));
    CHECK(cudaFree(sim->d_P.y));
    CHECK(cudaFree(sim->d_P.z));
    CHECK(cudaFree(sim->d_P.vx));
    CHECK(cudaFree(sim->d_P.vy));
    CHECK(cudaFree(sim->d_P.vz));
//    CHECK(cudaFree(sim->d_new_P));
    CHECK(cudaFree(sim->d_new_P.x));
    CHECK(cudaFree(sim->d_new_P.y));
    CHECK(cudaFree(sim->d_new_P.z));
    CHECK(cudaFree(sim->d_new_P.vx));
    CHECK(cudaFree(sim->d_new_P.vy));
    CHECK(cudaFree(sim->d_new_P.vz));

    CHECK(cudaFree(sim->d_samples));
    CHECK(cudaFree(sim->d_cellCount));
    CHECK(cudaFree(sim->d_cellCountPrefixSum));
    CHECK(cudaFree(sim->d_temp_storage));

    CHECK(cudaFree(sim->d_totalCollisions));
    CHECK(cudaFree(sim->d_new_NP));

    CHECK(cudaFree(sim->d_valid));
    CHECK(cudaFree(sim->d_particleIds));
    CHECK(cudaFree(sim->d_particleIdsSorted));
    CHECK(cudaFree(sim->d_cellKeys));
    CHECK(cudaFree(sim->d_cellKeysSorted));
}
