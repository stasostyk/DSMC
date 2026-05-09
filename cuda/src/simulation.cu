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
    sim->conf = (Config *) malloc(sizeof(Config));

    memcpy(sim->conf, conf, sizeof(Config));
    CHECK(cudaMemcpyToSymbol(d_conf, conf, sizeof(Config)));

    sim->P = (Particle *)malloc(PARTICLES_SZ);
    sim->samples = (Cell *)malloc(SAMPLES_SZ);

    CHECK(cudaMalloc(&sim->rngStates, MAX_PARTICLES * sizeof(curandState)));
    CHECK(cudaMalloc(&sim->d_P, PARTICLES_SZ));
    CHECK(cudaMalloc(&sim->d_new_P, PARTICLES_SZ));
    CHECK(cudaMalloc(&sim->d_samples, SAMPLES_SZ));
    CHECK(cudaMalloc(&sim->d_cellCount, CELL_COUNT_SZ));
    CHECK(cudaMalloc(&sim->d_cellList, CELL_LIST_SZ));

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
    Particle *P, int *cellCount, int *cellList, int NP
) {
    // TODO this seems like a bin pattern, could be improved
    // note: the bottleneck is the atomicAdd

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    int k = (int)(P[i].x / d_conf.dx);
    int l = (int)(P[i].y / d_conf.dy);
    int m = (int)(P[i].z / d_conf.dz);

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
    dim3 threadsPerBlock(4, 4, 4);
    dim3 blocksPerGrid(
        (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    // TODO use cuda_memset instead
    // TODO maybe this can be called from a different stream?
    reset_cell_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(sim->d_cellCount);
    CHECK_KERNELCALL();

    int thr = 64;
    dim3 threadsPerBlock2(thr, 1, 1);
    dim3 blocksPerGrid2((sim->NP + thr - 1) / thr, 1, 1);
    bin_particles_kernel<<<blocksPerGrid2, threadsPerBlock2>>>(
        sim->d_P, sim->d_cellCount, sim->d_cellList, sim->NP
    );
    CHECK_KERNELCALL();
}

__global__ void generate_particles_in_rect_kernel(
    Particle *P,
    int start,
    int Nnew,
    double x1, double x2, 
    double y1, double y2,
    double z1, double z2,
    int moveFlag,
    curandState *rngStates
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nnew) return;

    double ux = d_conf.UxFree;
    double uy = d_conf.UyFree;
    double uz = d_conf.UzFree; 
    double dt = d_conf.dt;

    int idx = start + i;

    curandState rngState = rngStates[idx];

    double rx = curand_uniform_double(&rngState);
    double ry = curand_uniform_double(&rngState);
    double rz = curand_uniform_double(&rngState);

    // TODO possibly better to create everything in local variable, and only then store in P?
    P[idx].x = x1 + (x2 - x1) * rx;
    P[idx].y = y1 + (y2 - y1) * ry;
    P[idx].z = z1 + (z2 - z1) * rz;

    double vx = curand_normal_double(&rngState) * d_conf.generation_derivatedMultiplier + ux;
    double vy = curand_normal_double(&rngState) * d_conf.generation_derivatedMultiplier + uy;
    double vz = curand_normal_double(&rngState) * d_conf.generation_derivatedMultiplier + uz;

    P[idx].vx = vx;
    P[idx].vy = vy;
    P[idx].vz = vz;

    if (moveFlag) {
        P[idx].x += dt * vx;
        P[idx].y += dt * vy;
        P[idx].z += dt * vz;
    }

    rngStates[idx] = rngState;
}


void generate_particles_in_rect(
    Simulation *sim,
    double x1, double x2,
    double y1, double y2,
    double z1, double z2,
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

#ifdef BALL_CASE
// TODO this kernel (and the function that calls it) 
// is super similar to filter_particles_out_of_bounds,
// maybe possible to use the same one for filtering.
__global__ void filter_particles_inside_ball(
    Particle *P, Particle *P_out, int NP, int *new_NP
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    double cx = d_conf.ballCenterX;
    double cy = d_conf.ballCenterY;
    double cz = d_conf.ballCenterZ;
    double R2 = d_conf.ballRadiusSquared;

    double rx = P[i].x - cx;
    double ry = P[i].y - cy;
    double rz = P[i].z - cz;

    // check if particle valid
    if (rx*rx + ry*ry + rz*rz > R2) {
        int pos = atomicAdd(new_NP, 1);
        P_out[pos] = P[i];
    }
}

void remove_particles_inside_ball(Simulation *sim) {
    int threads = 128;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((sim->NP + threads - 1) / threads, 1, 1);

    int *d_new_NP;
    CHECK(cudaMalloc(&d_new_NP, sizeof(int)));
    CHECK(cudaMemset(d_new_NP, 0, sizeof(int)));

    filter_particles_inside_ball<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_P, sim->d_new_P, sim->NP, d_new_NP
    );
    CHECK_KERNELCALL();

    int host_new_NP;
    CHECK(cudaMemcpy(&host_new_NP, d_new_NP, sizeof(int), cudaMemcpyDeviceToHost));
    sim->NP = host_new_NP;
    
    // swap d_P and d_new_P
    Particle *tmp = sim->d_P;
    sim->d_P = sim->d_new_P;
    sim->d_new_P = tmp;

    CHECK(cudaFree(d_new_NP));
}
#endif

void initialize_particles(Simulation *sim) {
    sim->NP = 0;
    generate_particles_in_rect(sim, 0.0, sim->conf->Lx, 0.0, sim->conf->Ly, 0.0, sim->conf->Lz, 0);
#ifdef BALL_CASE
    remove_particles_inside_ball(sim);
#endif
}

__global__ void filter_particles_out_of_bounds(
    Particle *P, Particle *P_out, int NP, int *new_NP
) {
    // TODO this can be possibly optimized:
    // 1st kernel: mark the particles true/false if they need to be removed
    // 2nd kernel: scan/prefix sum 
    // 3rd kernel: (scatter) collect valid particles

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    // check if particle valid
    if (P[i].x >= 0.0 && P[i].x < d_conf.Lx &&
        P[i].y >= 0.0 && P[i].y < d_conf.Ly &&
        P[i].z >= 0.0 && P[i].z < d_conf.Lz
    ) {
        int pos = atomicAdd(new_NP, 1);
        P_out[pos] = P[i];
    }
}

void apply_boundary_conditions_free_stream(Simulation *sim) {
    // TODO potentially could be done all in parallel?
    // TODO maybe the kernels to generate particles could be started even before
    //      the previous task (move_particles) is finished because generation
    //      could be done in a way so that it doesnt overlap? (note: use different streams)
    Config *conf = sim->conf;
    generate_particles_in_rect(sim, -(conf->DL), 0.0, 0.0, conf->Ly, 0.0, conf->Lz, 1);
    generate_particles_in_rect(sim, conf->Lx, conf->Lx + conf->DL, 0.0, conf->Ly, 0.0, conf->Lz, 1);
    generate_particles_in_rect(sim, 0.0, conf->Lx, -(conf->DL), 0.0, 0.0, conf->Lz, 1);
    generate_particles_in_rect(sim, 0.0, conf->Lx, conf->Ly, conf->Ly + conf->DL, 0.0, conf->Lz, 1);
    generate_particles_in_rect(sim, 0.0, conf->Lx, 0.0, conf->Ly, -(conf->DL), 0.0, 1);
    generate_particles_in_rect(sim, 0.0, conf->Lx, 0.0, conf->Ly, conf->Lz, conf->Lz + conf->DL, 1);

    int threads = 128;
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid((sim->NP + threads - 1) / threads, 1, 1);

    int *d_new_NP;
    CHECK(cudaMalloc(&d_new_NP, sizeof(int)));
    CHECK(cudaMemset(d_new_NP, 0, sizeof(int)));

    filter_particles_out_of_bounds<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_P, sim->d_new_P, sim->NP, d_new_NP
    );
    CHECK_KERNELCALL();

    int host_new_NP;
    CHECK(cudaMemcpy(&host_new_NP, d_new_NP, sizeof(int), cudaMemcpyDeviceToHost));
    sim->NP = host_new_NP;
    
    // swap d_P and d_new_P
    Particle *tmp = sim->d_P;
    sim->d_P = sim->d_new_P;
    sim->d_new_P = tmp;

    CHECK(cudaFree(d_new_NP));
}

__global__ void move_particles_kernel(Particle *P, int NP, curandState *rngStates) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    // TODO a lot of calls to global memory P[i], probably would be better to save to local

    double X0 = P[i].x;
    double Y0 = P[i].y;
    double Z0 = P[i].z;
    P[i].x += d_conf.dt * P[i].vx;
    P[i].y += d_conf.dt * P[i].vy;
    P[i].z += d_conf.dt * P[i].vz;

    curandState rngState = rngStates[i];


    #ifdef WING_CASE
    double dt = d_conf.dt;
    double moleculeMass = d_conf.moleculeMass;
    double Tw = d_conf.Tw;
    double WingX = d_conf.WingX;
    double WingY = d_conf.WingY;
    double WingLength = d_conf.WingLength;

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
                d_conf.KB,
                &rngState
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
        double cx = d_conf.ballCenterX;
        double cy = d_conf.ballCenterY;
        double cz = d_conf.ballCenterZ;
        double ballRadius = d_conf.ballRadius;

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
                double Dt1 = d_conf.dt * (1.0 - t_hit);

                // Surface normal (outward)
                double nx = (Xw - cx) / ballRadius;
                double ny = (Yw - cy) / ballRadius;
                double nz = (Zw - cz) / ballRadius;

                // Diffuse reflection aligned with normal
                diffuse_scattering_device(&(P[i].vx), &(P[i].vy), &(P[i].vz),
                                d_conf.moleculeMass, d_conf.Tb,
                                nx, ny, nz, d_conf.KB,
                                &rngState
                            );

                // Move after collision
                P[i].x = Xw + Dt1 * P[i].vx;
                P[i].y = Yw + Dt1 * P[i].vy;
                P[i].z = Zw + Dt1 * P[i].vz;
            }
        }
    }

    #endif

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

    dim3 threadsPerBlock(4, 4, 4);
    dim3 blocksPerGrid(
        (NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    accumulate_sampling_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        sim->d_cellCount, sim->d_cellList, sim->d_samples, sim->d_P
    );
    CHECK_KERNELCALL();
}

void clearPointers(Simulation *sim) {
    free(sim->conf);
    free(sim->P);
    free(sim->samples);

    CHECK(cudaFree(sim->rngStates));
    CHECK(cudaFree(sim->d_P));
    CHECK(cudaFree(sim->d_new_P));
    CHECK(cudaFree(sim->d_samples));
    CHECK(cudaFree(sim->d_cellCount));
    CHECK(cudaFree(sim->d_cellList));

    CHECK(cudaFree(sim->d_totalCollisions));
}
