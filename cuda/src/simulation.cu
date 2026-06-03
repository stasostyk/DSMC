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

void swap_particles_with_new(Simulation *sim) {
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
    sim->P.pos = (float *)malloc(PARTICLES_FIELD_SZ);
    sim->P.vel = (float *)malloc(PARTICLES_FIELD_SZ);


    sim->samples = (Cell *)malloc(SAMPLES_SZ);

    CHECK(cudaMalloc(&sim->rngStates, MAX_PARTICLES * sizeof(curandState)));
    
    // device SoA for particles
    CHECK(cudaMalloc(&sim->d_P.pos,  PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_P.vel,  PARTICLES_FIELD_SZ));
    
    // device SoA for new particles
    CHECK(cudaMalloc(&sim->d_new_P.pos,  PARTICLES_FIELD_SZ));
    CHECK(cudaMalloc(&sim->d_new_P.vel,  PARTICLES_FIELD_SZ));

    CHECK(cudaMalloc(&sim->d_samples, SAMPLES_SZ));
    CHECK(cudaMalloc(&sim->d_cellCount, CELL_COUNT_SZ));
    CHECK(cudaMalloc(&sim->d_cellCountPrefixSum, CELL_COUNT_SZ));

    CHECK(cudaMalloc(&sim->d_valid, sizeof(int) * MAX_PARTICLES));
    CHECK(cudaMalloc(&sim->d_particleIds, sizeof(int) * MAX_PARTICLES));
    CHECK(cudaMalloc(&sim->d_cellKeys, sizeof(int) * MAX_PARTICLES));
    CHECK(cudaMalloc(&sim->d_cellCountPrefSumCopy, max(CELL_COUNT_SZ, sizeof(int) * MAX_PARTICLES)));

    sim->temp_storage_bytes = 0;
    sim->d_temp_storage = nullptr;

    cub::DeviceScan::ExclusiveSum(
        sim->d_temp_storage, 
        sim->temp_storage_bytes, 
        sim->d_cellCount, 
        sim->d_cellCountPrefixSum, 
        NX*NY*NZ);
    cudaMalloc(&sim->d_temp_storage, sim->temp_storage_bytes);

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

    float x = x1 + (x2 - x1) * rx;
    float y = y1 + (y2 - y1) * ry;
    float z = z1 + (z2 - z1) * rz;

    float vx = curand_normal(&rngState) * d_conf.generation_derivatedMultiplier + ux;
    float vy = curand_normal(&rngState) * d_conf.generation_derivatedMultiplier + uy;
    float vz = curand_normal(&rngState) * d_conf.generation_derivatedMultiplier + uz;

    P.vel[IDX_PARTICLE(idx, 0)] = vx;
    P.vel[IDX_PARTICLE(idx, 1)] = vy;
    P.vel[IDX_PARTICLE(idx, 2)] = vz;

    if (moveFlag) {
        x += dt * vx;
        y += dt * vy;
        z += dt * vz;
    }
    
    P.pos[IDX_PARTICLE(idx, 0)] = x;
    P.pos[IDX_PARTICLE(idx, 1)] = y;
    P.pos[IDX_PARTICLE(idx, 2)] = z;

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

    float x = P.pos[IDX_PARTICLE(i, 0)];
    float y = P.pos[IDX_PARTICLE(i, 1)];
    float z = P.pos[IDX_PARTICLE(i, 2)];

    valid[i] =
        (x >= 0.0f && x < d_conf.Lx &&
         y >= 0.0f && y < d_conf.Ly &&
         z >= 0.0f && z < d_conf.Lz);
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

        float rx = P.pos[IDX_PARTICLE(i, 0)] - cx;
        float ry = P.pos[IDX_PARTICLE(i, 1)] - cy;
        float rz = P.pos[IDX_PARTICLE(i, 2)] - cz;

        insideBall |= (rx*rx + ry*ry + rz*rz <= R2);
    }

    if (!insideBall) {
        int pos = atomicAdd(new_NP, 1);
        P_out.pos[IDX_PARTICLE(pos, 0)] = P.pos[IDX_PARTICLE(i, 0)];
        P_out.pos[IDX_PARTICLE(pos, 1)] = P.pos[IDX_PARTICLE(i, 1)];
        P_out.pos[IDX_PARTICLE(pos, 2)] = P.pos[IDX_PARTICLE(i, 2)];
        P_out.vel[IDX_PARTICLE(pos, 0)] = P.vel[IDX_PARTICLE(i, 0)];
        P_out.vel[IDX_PARTICLE(pos, 1)] = P.vel[IDX_PARTICLE(i, 1)];
        P_out.vel[IDX_PARTICLE(pos, 2)] = P.vel[IDX_PARTICLE(i, 2)];
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

    float x = P.pos[IDX_PARTICLE(i, 0)];
    float y = P.pos[IDX_PARTICLE(i, 1)];
    float z = P.pos[IDX_PARTICLE(i, 2)];

    // write compact particle
    P_out.pos[IDX_PARTICLE(j, 0)]  = x;
    P_out.pos[IDX_PARTICLE(j, 1)]  = y;
    P_out.pos[IDX_PARTICLE(j, 2)]  = z;
    P_out.vel[IDX_PARTICLE(j, 0)] = P.vel[IDX_PARTICLE(i, 0)];
    P_out.vel[IDX_PARTICLE(j, 1)] = P.vel[IDX_PARTICLE(i, 1)];
    P_out.vel[IDX_PARTICLE(j, 2)] = P.vel[IDX_PARTICLE(i, 2)];

    // compute cell key
    int k = __float2int_rd(x * d_conf.inv_dx);
    int l = __float2int_rd(y * d_conf.inv_dy);
    int m = __float2int_rd(z * d_conf.inv_dz);

    int cell = IDX_CELL(k,l,m);
    cellKeys[j] = cell;
    atomicAdd(&cellCount[cell], 1);
}

__global__ void counting_sort_scatter_kernel(
    Particles P_in, Particles P_out,
    int *cellKeys,           // cell of each compacted particle
    int *cellOffsets,        // prefix sum of cellCount (write cursor per cell)
    int NP
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    int cell = cellKeys[i];
    int dest = atomicAdd(&cellOffsets[cell], 1);  // claim a slot

    P_out.pos[IDX_PARTICLE(dest, 0)]  = P_in.pos[IDX_PARTICLE(i, 0)];
    P_out.pos[IDX_PARTICLE(dest, 1)]  = P_in.pos[IDX_PARTICLE(i, 1)];
    P_out.pos[IDX_PARTICLE(dest, 2)]  = P_in.pos[IDX_PARTICLE(i, 2)];
    P_out.vel[IDX_PARTICLE(dest, 0)] = P_in.vel[IDX_PARTICLE(i, 0)];
    P_out.vel[IDX_PARTICLE(dest, 1)] = P_in.vel[IDX_PARTICLE(i, 1)];
    P_out.vel[IDX_PARTICLE(dest, 2)] = P_in.vel[IDX_PARTICLE(i, 2)];
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

    cub::DeviceScan::ExclusiveSum(
        sim->d_temp_storage, sim->temp_storage_bytes,
        sim->d_cellCount, sim->d_cellCountPrefixSum,
        NX * NY * NZ
    );

    // Copy prefix sum into a mutable "cursor" array (reuse d_particleIds as scratch)
    CHECK(cudaMemcpy(sim->d_cellCountPrefSumCopy, sim->d_cellCountPrefixSum,
            sizeof(int) * NX * NY * NZ, cudaMemcpyDeviceToDevice));

    counting_sort_scatter_kernel<<<blocksNew, threadsPerBlock>>>(
        sim->d_new_P, sim->d_P,
        sim->d_cellKeys,
        sim->d_cellCountPrefSumCopy,   // cursor (gets incremented)
        h_newNP
    );
    CHECK_KERNELCALL();

    sim->NP = h_newNP;
}

__global__ void move_particles_kernel(Particles P, int NP, curandState *rngStates) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    // initial positions
    float x0 = P.pos[IDX_PARTICLE(i, 0)];
    float y0 = P.pos[IDX_PARTICLE(i, 1)];
    float z0 = P.pos[IDX_PARTICLE(i, 2)];

    // final positions
    // move this timestamp
    float x1 = x0 + d_conf.dt * P.vel[IDX_PARTICLE(i, 0)];
    float y1 = y0 + d_conf.dt * P.vel[IDX_PARTICLE(i, 1)];
    float z1 = z0 + d_conf.dt * P.vel[IDX_PARTICLE(i, 2)];

    curandState rngState = rngStates[i];

    // CHECK WINGS
    for (int wingId = 0; wingId < d_conf.wingCnt; wingId++) {
        float WingY = d_conf.wings[wingId].WingY;

        if ( ( y0 - WingY ) * ( y1 - WingY ) < 0.0 ) {
            float dt = d_conf.dt;
            double moleculeMass = d_conf.moleculeMass;
            double Tw = d_conf.wings[wingId].Tw;
            float WingX = d_conf.wings[wingId].WingX;
            float WingLength = d_conf.wings[wingId].WingLength;
            // Linear interpolation to point Y = WingY
            float Xw=( x0*(WingY-y1)+x1*(y0-WingY))/(y0-y1);
            float Zw=( z0*(WingY-y1)+z1*(y0-WingY))/(y0-y1);
            if ( Zw < 0.3 || Zw > 0.7 ) continue; // wing only occupies 0.3 < z < 0.7
            if ( Xw > WingX && Xw < WingX + WingLength ) {
                // Molecule interacts with the wing during the time step
                // Linear interpolation of the time of scattering, Eq. (6.5.4)
                float Dt1 = dt - dt * ( y0 - WingY ) / ( y0 - y1 );
                // Generate velocity vector of the reflected molecule
                diffuse_scattering_y_device(
                    &(P.vel[IDX_PARTICLE(i, 0)]), &(P.vel[IDX_PARTICLE(i, 1)]), &(P.vel[IDX_PARTICLE(i, 2)]),
                    moleculeMass,Tw,(y0-WingY>0)?1.0:(-1.0),
                    d_conf.KB,
                    &rngState
                );
                // Move the reflected molecule
                x1 = Xw + Dt1 * P.vel[IDX_PARTICLE(i, 0)];
                y1 = WingY + Dt1 * P.vel[IDX_PARTICLE(i, 1)];
            }
        }
    }

    // CHECK BALLS
    for (int ballId = 0; ballId < d_conf.ballCnt; ballId++) {
        // Ray-sphere intersection test
        
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
                diffuse_scattering_device(&(P.vel[IDX_PARTICLE(i, 0)]), &(P.vel[IDX_PARTICLE(i, 1)]), &(P.vel[IDX_PARTICLE(i, 2)]),
                                d_conf.moleculeMass, d_conf.balls[ballId].Tb,
                                nx, ny, nz, d_conf.KB,
                                &rngState
                            );

                // Move after collision
                x1 = Xw + Dt1 * P.vel[IDX_PARTICLE(i, 0)];
                y1 = Yw + Dt1 * P.vel[IDX_PARTICLE(i, 1)];
                z1 = Zw + Dt1 * P.vel[IDX_PARTICLE(i, 2)];
            }
        }
    }

    // save final positions
    P.pos[IDX_PARTICLE(i, 0)] = x1;
    P.pos[IDX_PARTICLE(i, 1)] = y1;
    P.pos[IDX_PARTICLE(i, 2)] = z1;

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
        int i = offset + q;

        float vx = P.vel[IDX_PARTICLE(i, 0)];
        float vy = P.vel[IDX_PARTICLE(i, 1)];
        float vz = P.vel[IDX_PARTICLE(i, 2)];

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
    free(sim->P.pos);
    free(sim->P.vel);
    free(sim->samples);

    CHECK(cudaFree(sim->rngStates));

    CHECK(cudaFree(sim->d_P.pos));
    CHECK(cudaFree(sim->d_P.vel));
    CHECK(cudaFree(sim->d_new_P.pos));
    CHECK(cudaFree(sim->d_new_P.vel));

    CHECK(cudaFree(sim->d_samples));
    CHECK(cudaFree(sim->d_cellCount));
    CHECK(cudaFree(sim->d_cellCountPrefixSum));
    CHECK(cudaFree(sim->d_temp_storage));

    CHECK(cudaFree(sim->d_totalCollisions));
    CHECK(cudaFree(sim->d_new_NP));

    CHECK(cudaFree(sim->d_valid));
    CHECK(cudaFree(sim->d_particleIds));
    CHECK(cudaFree(sim->d_cellKeys));
    CHECK(cudaFree(sim->d_cellCountPrefSumCopy));
}
