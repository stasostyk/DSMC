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

__global__ void init_rng_kernel(curandState *states, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAX_PARTICLES) return;
    curand_init(seed, i, 0, &states[i]);
}

void setup(Simulation *sim, Config *conf) {
    sim->P = (Particle *)malloc(PARTICLES_SZ);
    sim->samples = (Cell *)malloc(SAMPLES_SZ);
    // sim->cellCount = (int *)malloc(CELL_COUNT_SZ);
    // sim->cellList = (int *)malloc(CELL_LIST_SZ);

    CHECK(cudaMalloc(&sim->rngStates, MAX_PARTICLES * sizeof(curandState)));
    CHECK(cudaMalloc(&sim->d_P, PARTICLES_SZ));
    CHECK(cudaMalloc(&sim->d_samples, SAMPLES_SZ));
    CHECK(cudaMalloc(&sim->d_cellCount, CELL_COUNT_SZ));
    CHECK(cudaMalloc(&sim->d_cellList, CELL_LIST_SZ));

    sim->sampleSteps = 0;
    sim->NP = 0;
    sim->totalCollisions = 0;

    // TODO: most of the setup can be moved to constexpr??

    // memset(sim->samples, 0, sizeof(sim->samples));
    cudaMemset(sim->d_samples, 0, SAMPLES_SZ);

    sim->dx = conf->Lx / NX;
    sim->dy = conf->Ly / NY;
    sim->dz = conf->Lz / NZ;
    sim->cellVolume = sim->dx * sim->dy * sim->dz;

    sim->NP = NX * NY * NZ * conf->particlesPerCellTarget;

    if (sim->NP > MAX_PARTICLES) {
        fprintf(stderr, "Too many particles for MAX_PARTICLES\n");
        exit(1);
    }

    #ifdef WING_CASE
        double angleOfAttack = conf->angleOfAttack;
    #elif defined(BALL_CASE)
        double angleOfAttack = 0.;
    #endif
    sim->NFree = conf->PFree / ( conf->KB * conf->TFree );
    sim->UFree = conf->MaFree * sqrt ( ( 5.0 / 3.0 ) * conf->KB * conf->TFree / conf->moleculeMass );
    sim->UxFree = sim->UFree * cos ( M_PI * angleOfAttack / 180.0 );
    sim->UyFree = - sim->UFree * sin ( M_PI * angleOfAttack / 180.0 );
    sim->UzFree = 0.0;

    sim->weight = sim->NFree * sim->cellVolume / conf->particlesPerCellTarget;

    // init randomness
    int threads = 256;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((MAX_PARTICLES + threads - 1) / threads, 1, 1);

    // TODO put proper seed, as before with CPU
    init_rng_kernel<<<blocksPerGrid, threadsPerBlock>>>(sim->rngStates, 1234);
    CHECK_KERNELCALL();
}

__global__ void reset_cell_count_kernel(int *cellCount) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z * blockDim.z + threadIdx.z;

    if (k >= NX || l >= NY || m >= NZ) return;

    cellCount[IDX_CELL(k, l, m)] = 0;
}

// TODO dx, dy, dz could be constant memory (or in #define)
__global__ void bin_particles_kernel(
    Particle *P, int *cellCount, int *cellList, int NP,
    double dx, double dy, double dz
) {
    // TODO this seems like a bin pattern, could be improved
    // note: the bottleneck is the atomicAdd

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    int k = (int)(P[i].x / dx);
    int l = (int)(P[i].y / dy);
    int m = (int)(P[i].z / dz);

    if (k < 0) k = 0;
    if (k >= NX) k = NX - 1;
    if (l < 0) l = 0;
    if (l >= NY) l = NY - 1;
    if (m < 0) m = 0;
    if (m >= NZ) m = NZ - 1;

    int n = atomicAdd(&cellCount[IDX_CELL(k, l, m)], 1);
    cellList[IDX_LIST(k, l, m, n)] = i;
}

void index_particles(Simulation *sim) {
    // CHECK(cudaMemcpy(sim->d_cellCount, sim->cellCount, CELL_COUNT_SZ, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(sim->d_cellList, sim->cellList, CELL_LIST_SZ, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(sim->d_P, sim->P, PARTICLES_SZ, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid(
        (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    // TODO use cuda_memset instead
    reset_cell_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(sim->d_cellCount);
    CHECK_KERNELCALL();

    CHECK(cudaDeviceSynchronize());

    int thr = 128;
    dim3 threadsPerBlock2(thr, 1, 1);
    dim3 blocksPerGrid2((sim->NP + thr - 1) / thr, 1, 1);
    bin_particles_kernel<<<blocksPerGrid2, threadsPerBlock2>>>(
        sim->d_P, sim->d_cellCount, sim->d_cellList, sim->NP,
        sim->dx, sim->dy, sim->dz
    );
    CHECK_KERNELCALL();

    // CHECK(cudaMemcpy(sim->cellCount, sim->d_cellCount, CELL_COUNT_SZ, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(sim->cellList, sim->d_cellList, CELL_LIST_SZ, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(sim->P, sim->d_P, PARTICLES_SZ, cudaMemcpyDeviceToHost));
}

__global__ void generate_particles_in_rect_kernel(
    Particle *P,
    Config *conf,
    int start,
    int Nnew,
    double x1, double x2, 
    double y1, double y2,
    double z1, double z2,
    double ux, double uy, double uz,
    double dt,
    double Tgas,
    int moveFlag,
    curandState *rngStates
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nnew) return;

    int idx = start + i;
    // TODO rngStates create a local variable -- don't access the global one

    double rx = curand_uniform(&rngStates[idx]);
    double ry = curand_uniform(&rngStates[idx]);
    double rz = curand_uniform(&rngStates[idx]);

    // TODO possibly better to create everything in local variable, and only then store in P?
    P[idx].x = x1 + (x2 - x1) * rx;
    P[idx].y = y1 + (y2 - y1) * ry;
    P[idx].z = z1 + (z2 - z1) * rz;

    // TODO compute the sqrt in CPU, and pass it to GPU as param
    double vx = curand_normal(&rngStates[idx]) * sqrt(conf->KB * Tgas / conf->moleculeMass) + ux;
    double vy = curand_normal(&rngStates[idx]) * sqrt(conf->KB * Tgas / conf->moleculeMass) + uy;
    double vz = curand_normal(&rngStates[idx]) * sqrt(conf->KB * Tgas / conf->moleculeMass) + uz;

    P[idx].vx = vx;
    P[idx].vy = vy;
    P[idx].vz = vz;

    if (moveFlag) {
        P[idx].x += dt * vx;
        P[idx].y += dt * vy;
        P[idx].z += dt * vz;
    }
}


void generate_particles_in_rect(
    Simulation *sim,
    Config *conf,
    double x1, double x2,
    double y1, double y2,
    double z1, double z2,
    double Tgas,
    int moveFlag
) {
    double ngas = sim->NFree;
    double ux = sim->UxFree;
    double uy = sim->UyFree;
    double uz = sim->UzFree; 

    double V = (x2 - x1) * (y2 - y1) * (z2 - z1);
    double N_add_exp = ngas * V / sim->weight;
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

    int threads = 256;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((Nnew + threads - 1) / threads, 1, 1);

    Config *d_C;
    CHECK(cudaMalloc(&d_C, sizeof(Config)));

    // TODO: probably can be optimized that in simulation loop not always the cuda memcpy is needed
    // (takes a lot of time)
    // CHECK(cudaMemcpy(sim->d_P, sim->P, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, conf, sizeof(Config), cudaMemcpyHostToDevice));

    generate_particles_in_rect_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_P, d_C, sim->NP, Nnew, 
        x1, x2, y1, y2, z1, z2, ux, uy, uz, 
        conf->dt, Tgas, moveFlag, sim->rngStates
    );
    CHECK_KERNELCALL();

    // TODO copying is probably only required for sim->NP particles
    // CHECK(cudaMemcpy(sim->P, sim->d_P, MAX_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_C));

    sim->NP += Nnew;
}

void initialize_particles(Simulation *sim, Config *conf) {
    sim->NP = 0;
    generate_particles_in_rect(sim, conf, 0.0, conf->Lx, 0.0, conf->Ly, 0.0, conf->Lz, conf->TFree, 0);
}

__global__ void filter_particles_out_of_bounds(
    Config *conf, Particle *P, Particle *P_out, int NP, int *new_NP
) {
    // TODO this can be possibly optimized:
    // 1st kernel: mark the particles true/false if they need to be removed
    // 2nd kernel: scan/prefix sum 
    // 3rd kernel: (scatter) collect valid particles

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    // check if particle valid
    if (P[i].x >= 0.0 && P[i].x < conf->Lx &&
        P[i].y >= 0.0 && P[i].y < conf->Ly &&
        P[i].z >= 0.0 && P[i].z < conf->Lz
    ) {
        int pos = atomicAdd(new_NP, 1);
        P_out[pos] = P[i];
    }
}

void apply_boundary_conditions_free_stream(Simulation *sim, Config *conf) {
    // TODO potentially could be done all in parallel?
    // TODO maybe the kernels to generate particles could be started even before
    //      the previous task (move_particles) is finished because generation
    //      could be done in a way so that it doesnt overlap? (note: use different streams)
    generate_particles_in_rect(sim, conf, -(conf->DL), 0.0, 0.0, conf->Ly, 0.0, conf->Lz, conf->TFree, 1);
    generate_particles_in_rect(sim, conf, conf->Lx, conf->Lx + conf->DL, 0.0, conf->Ly, 0.0, conf->Lz, conf->TFree, 1);
    generate_particles_in_rect(sim, conf, 0.0, conf->Lx, -(conf->DL), 0.0, 0.0, conf->Lz, conf->TFree, 1);
    generate_particles_in_rect(sim, conf, 0.0, conf->Lx, conf->Ly, conf->Ly + conf->DL, 0.0, conf->Lz, conf->TFree, 1);
    generate_particles_in_rect(sim, conf, 0.0, conf->Lx, 0.0, conf->Ly, -(conf->DL), 0.0, conf->TFree, 1);
    generate_particles_in_rect(sim, conf, 0.0, conf->Lx, 0.0, conf->Ly, conf->Lz, conf->Lz + conf->DL, conf->TFree, 1);

    int threads = 256;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((sim->NP + threads - 1) / threads, 1, 1);

    Config *d_C;
    CHECK(cudaMalloc(&d_C, sizeof(Config)));

    int *d_new_NP;
    Particle *d_new_P;
    CHECK(cudaMalloc(&d_new_NP, sizeof(int)));
    CHECK(cudaMalloc(&d_new_P, sim->NP * sizeof(Particle)));
    CHECK(cudaMemset(d_new_NP, 0, sizeof(int)));

    // CHECK(cudaMemcpy(sim->d_P, sim->P, PARTICLES_SZ, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, conf, sizeof(Config), cudaMemcpyHostToDevice));

    filter_particles_out_of_bounds<<<blocksPerGrid, threadsPerBlock>>>(
        d_C, sim->d_P, d_new_P, sim->NP, d_new_NP
    );
    CHECK_KERNELCALL();

    int host_new_NP;
    CHECK(cudaMemcpy(&host_new_NP, d_new_NP, sizeof(int), cudaMemcpyDeviceToHost));
    sim->NP = host_new_NP;
    
    // CHECK(cudaMemcpy(sim->P, d_new_P, sim->NP * sizeof(Particle), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sim->d_P, d_new_P, sim->NP * sizeof(Particle), cudaMemcpyDeviceToDevice));

    CHECK(cudaFree(d_new_NP));
    CHECK(cudaFree(d_new_P));
    CHECK(cudaFree(d_C));
}

__global__ void move_particles_kernel(Particle *P, Config *conf, int NP, curandState *rngStates) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    // TODO a lot of calls to global memory P[i], probably would be better to save to local

    double X0 = P[i].x;
    double Y0 = P[i].y;
    double Z0 = P[i].z;
    P[i].x += conf->dt * P[i].vx;
    P[i].y += conf->dt * P[i].vy;
    P[i].z += conf->dt * P[i].vz;


    #ifdef WING_CASE
    double dt = conf->dt;
    double moleculeMass = conf->moleculeMass;
    double Tw = conf->Tw;
    double WingX = conf->WingX;
    double WingY = conf->WingY;
    double WingLength = conf->WingLength;

    if ( ( Y0 - WingY ) * ( P[i].y - WingY ) < 0.0 ) {
        // Linear interpolation to point Y = WingY
        double Xw=( X0*(WingY-P[i].y)+P[i].x*(Y0-WingY))/(Y0-P[i].y);
        double Zw=( Z0*(WingY-P[i].y)+P[i].z*(Y0-WingY))/(Y0-P[i].y);
        if ( Zw < 0.3 || Zw > 0.7 ) return; // wing only occupies 0.3 < z < 0.7
        if ( Xw > WingX && Xw < WingX + WingLength ) {
            // Molecule interacts with the wing during the time step
            // Linear interpolation of the time of scattering, Eq. (6.5.4)
            double Dt1 = dt - dt * ( Y0 - WingY ) / ( Y0 - P[i].y );
            // Generate velocity vector of the reflected molecule
            diffuse_scattering_y_device(
                &(P[i].vx), &(P[i].vy), &(P[i].vz), 
                moleculeMass,Tw,(Y0-WingY>0)?1.0:(-1.0),
                conf->KB,
                &rngStates[i]
            );
            // Move the reflected molecule
            P[i].x = Xw + Dt1 * P[i].vx;
            P[i].y = WingY + Dt1 * P[i].vy;
        }
    }
    #elif defined(BALL_CASE)
    // Ray-sphere intersection test
    {
        // Initial and final positions
        double x0 = X0, y0 = Y0, z0 = Z0;
        double x1 = P[i].x, y1 = P[i].y, z1 = P[i].z;

        // Direction of motion
        double dx = x1 - x0;
        double dy = y1 - y0;
        double dz = z1 - z0;

        // Sphere center
        double cx = conf->ballCenterX;
        double cy = conf->ballCenterY;
        double cz = conf->ballCenterZ;
        double ballRadius = conf->ballRadius;

        // Shifted initial position
        double rx = x0 - cx;
        double ry = y0 - cy;
        double rz = z0 - cz;

        // Quadratic coefficients: |r + t d|^2 = R^2
        double a = dx*dx + dy*dy + dz*dz;
        double b = 2.0 * (rx*dx + ry*dy + rz*dz);
        double c = rx*rx + ry*ry + rz*rz - ballRadius*ballRadius;

        double disc = b*b - 4.0*a*c;

        if (disc >= 0.0) {
            double sqrt_disc = sqrt(disc);

            // time solutions
            double t1 = (-b - sqrt_disc) / (2.0*a);
            double t2 = (-b + sqrt_disc) / (2.0*a);

            // pick earliest valid intersection in [0,1]
            double t_hit = -1.0;
            if (t1 >= 0.0 && t1 <= 1.0) t_hit = t1;
            else if (t2 >= 0.0 && t2 <= 1.0) t_hit = t2;

            if (t_hit >= 0.0) {
                // Intersection point
                double Xw = x0 + t_hit * dx;
                double Yw = y0 + t_hit * dy;
                double Zw = z0 + t_hit * dz;

                // Remaining time after collision
                double Dt1 = conf->dt * (1.0 - t_hit);

                // Surface normal (outward)
                double nx = (Xw - cx) / ballRadius;
                double ny = (Yw - cy) / ballRadius;
                double nz = (Zw - cz) / ballRadius;

                // Diffuse reflection aligned with normal
                diffuse_scattering_device(&(P[i].vx), &(P[i].vy), &(P[i].vz),
                                conf->moleculeMass, conf->Tb,
                                nx, ny, nz, conf->KB,
                                &rngStates[i]
                            );

                // Move after collision
                P[i].x = Xw + Dt1 * P[i].vx;
                P[i].y = Yw + Dt1 * P[i].vy;
                P[i].z = Zw + Dt1 * P[i].vz;
            }
        }
    }

    #endif

}

void move_particles(Simulation *sim, Config *conf) {
    int threads = 256;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((sim->NP + threads - 1) / threads, 1, 1);

    // TODO reuse this so that it wouldn't be needed to malloc and free and copy everytime
    Config *d_C;
    CHECK(cudaMalloc(&d_C, sizeof(Config)));

    // TODO: probably can be optimized that in simulation loop not always the cuda memcpy is needed
    // (takes a lot of time)
    // CHECK(cudaMemcpy(sim->d_P, sim->P, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, conf, sizeof(Config), cudaMemcpyHostToDevice));

    move_particles_kernel<<<blocksPerGrid, threadsPerBlock>>>(sim->d_P, d_C, sim->NP, sim->rngStates);
    CHECK_KERNELCALL();

    // CHECK(cudaMemcpy(sim->P, sim->d_P, MAX_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_C));
}

__global__ void accumulate_sampling_kernel(
    int *cellCount, int *cellList, Cell *samples, Particle *P
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z * blockDim.z + threadIdx.z;

    if (k >= NX || l >= NY || m >= NZ) return;

    int Nc = cellCount[IDX_CELL(k, l, m)];

    samples[IDX_CELL(k, l, m)].countNP += Nc;

    for (int q = 0; q < Nc; q++) {
        int i = cellList[IDX_LIST(k, l, m, q)];

        double vx = P[i].vx;
        double vy = P[i].vy;
        double vz = P[i].vz;

        samples[IDX_CELL(k, l, m)].countVx += vx;
        samples[IDX_CELL(k, l, m)].countVy += vy;
        samples[IDX_CELL(k, l, m)].countVz += vz;
        samples[IDX_CELL(k, l, m)].countV2 += vx * vx + vy * vy + vz * vz;
    }
}


void accumulate_sampling(Simulation *sim) {
    sim->sampleSteps++;

    // CHECK(cudaMemcpy(sim->d_cellCount, sim->cellCount, CELL_COUNT_SZ, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(sim->d_cellList, sim->cellList, CELL_LIST_SZ, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(sim->d_P, sim->P, PARTICLES_SZ, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(sim->d_samples, sim->samples, SAMPLES_SZ, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid(
        (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    accumulate_sampling_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_cellCount, sim->d_cellList, sim->d_samples, sim->d_P
    );
    CHECK_KERNELCALL();

    // CHECK(cudaMemcpy(sim->cellCount, sim->d_cellCount, CELL_COUNT_SZ, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(sim->cellList, sim->d_cellList, CELL_LIST_SZ, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(sim->P, sim->d_P, PARTICLES_SZ, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(sim->samples, sim->d_samples, SAMPLES_SZ, cudaMemcpyDeviceToHost));
}

void clearPointers(Simulation *sim) {
    free(sim->P);
    free(sim->samples);
    // free(sim->cellCount);
    // free(sim->cellList);

    CHECK(cudaFree(sim->rngStates));
    CHECK(cudaFree(sim->d_P));
    CHECK(cudaFree(sim->d_samples));
    CHECK(cudaFree(sim->d_cellCount));
    CHECK(cudaFree(sim->d_cellList));
}
