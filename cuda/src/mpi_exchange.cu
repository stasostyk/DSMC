#include "../include/mpi_exchange.h"
#include <cuda_runtime.h>
#include "../include/particles.h"
#include "../include/mpi_helper.h"
#include "../include/simulation.h"
#include "../include/config.h"
#include <cub/cub.cuh>
#include "../include/cuda_utils.h"

#include <thrust/iterator/transform_iterator.h>

struct FlagEquals {
    int target;
    __device__ int operator()(int f) const { return f == target; }
};

__global__ void classify_particles_kernel(
    Particles P, int NP,
    float x_lo, float x_hi,
    int *d_flag,   // 0=keep, 1=send_left, 2=send_right
    int *d_count   // [0]=keep, [1]=send_left, [2]=send_right (atomics)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    float x = P.x[i];
    int flag;

    if (x < x_lo) 
        flag = 1;
    else if (x >= x_hi) 
        flag = 2;
    else 
        flag = 0;

    d_flag[i] = flag;
    atomicAdd(&d_count[flag], 1);
}

__global__ void scatter_send_kernel(
    Particles P, int NP,
    int *d_flag, int *d_prefix_left, int *d_prefix_right,  // prefix sum of (flag==1) and (flag==2)
    float *send_left,            // packed: x,y,z,vx,vy,vz interleaved
    float *send_right,
    int *d_keep_prefix,          // prefix sum of (flag==0)
    Particles P_keep             // compacted in-place output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NP) return;

    int flag = d_flag[i];

    if (flag == 1) {
        int j = d_prefix_left[i] * 6;  // prefix of send_left flags
        send_left[j+0] = P.x[i];  send_left[j+1] = P.y[i];
        send_left[j+2] = P.z[i];  send_left[j+3] = P.vx[i];
        send_left[j+4] = P.vy[i]; send_left[j+5] = P.vz[i];
    } else if (flag == 2) {
        int j = d_prefix_right[i] * 6;  // prefix of send_right flags
        send_right[j+0] = P.x[i];  send_right[j+1] = P.y[i];
        send_right[j+2] = P.z[i];  send_right[j+3] = P.vx[i];
        send_right[j+4] = P.vy[i]; send_right[j+5] = P.vz[i];
    } else {
        int j = d_keep_prefix[i];
        P_keep.x[j]  = P.x[i]; P_keep.y[j]  = P.y[i]; P_keep.z[j]  = P.z[i];
        P_keep.vx[j] = P.vx[i]; P_keep.vy[j] = P.vy[i]; P_keep.vz[j] = P.vz[i];
    }
}

void swap_particles_with_new_another_func(Simulation *sim) {
    Particles temp = sim->d_P;
    sim->d_P = sim->d_new_P;
    sim->d_new_P = temp;
}

__global__ void unpack_recv_kernel(
    Particles P, int offset,
    float *recv_buf, int recv_n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= recv_n) return;

    int j = offset + i;
    P.x[j]  = recv_buf[i*6+0]; P.y[j]  = recv_buf[i*6+1];
    P.z[j]  = recv_buf[i*6+2]; P.vx[j] = recv_buf[i*6+3];
    P.vy[j] = recv_buf[i*6+4]; P.vz[j] = recv_buf[i*6+5];
}

void exchange_boundary_particles(Simulation *sim, MPIHelper *mpiHelper) {
    int threads = 128;
    dim3 block(threads);
    dim3 grid((sim->NP + threads - 1) / threads);

    CHECK(cudaMemset(mpiHelper->d_count, 0, 3 * sizeof(int)));

    classify_particles_kernel<<<grid, block>>>(
        sim->d_P, sim->NP, mpiHelper->xMin, mpiHelper->xMax,
        mpiHelper->d_flag, mpiHelper->d_count
    );
    CHECK_KERNELCALL();

    auto is_left  = thrust::make_transform_iterator(mpiHelper->d_flag, FlagEquals{1});
    auto is_right = thrust::make_transform_iterator(mpiHelper->d_flag, FlagEquals{2});
    auto is_keep  = thrust::make_transform_iterator(mpiHelper->d_flag, FlagEquals{0});

    cub::DeviceScan::ExclusiveSum(sim->d_temp_storage, sim->temp_storage_bytes,
        is_left,  mpiHelper->d_prefix_left,  sim->NP);
    cub::DeviceScan::ExclusiveSum(sim->d_temp_storage, sim->temp_storage_bytes,
        is_right, mpiHelper->d_prefix_right, sim->NP);
    cub::DeviceScan::ExclusiveSum(sim->d_temp_storage, sim->temp_storage_bytes,
        is_keep,  mpiHelper->d_prefix_keep,  sim->NP);


    int h_count[3];
    CHECK(cudaMemcpy(h_count, mpiHelper->d_count, 3 * sizeof(int), cudaMemcpyDeviceToHost));
    int keep_n  = h_count[0];
    int left_n  = h_count[1];
    int right_n = h_count[2];


    scatter_send_kernel<<<grid, block>>>(
        sim->d_P, sim->NP,
        mpiHelper->d_flag,
        mpiHelper->d_prefix_left,   // used for send_left indexing
        mpiHelper->d_prefix_right,  // used for send_right indexing  
        mpiHelper->d_send_left_mapped,
        mpiHelper->d_send_right_mapped,
        mpiHelper->d_prefix_keep,
        sim->d_new_P                // compacted survivors go here
    );
    CHECK_KERNELCALL();

    //    // --- Copy send buffers device -> host ---
    // CHECK(cudaMemcpy(mpiHelper->h_send_left,  mpiHelper->d_send_left,
    //                  left_n  * 6 * sizeof(float), cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(mpiHelper->h_send_right, mpiHelper->d_send_right,
    //                  right_n * 6 * sizeof(float), cudaMemcpyDeviceToHost));


    // --- Exchange counts ---
    int recv_left_n = 0, recv_right_n = 0;
    MPI_Sendrecv(&left_n,       1, MPI_INT, mpiHelper->left_rank,  0,
                 &recv_right_n, 1, MPI_INT, mpiHelper->right_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&right_n,      1, MPI_INT, mpiHelper->right_rank, 1,
                 &recv_left_n,  1, MPI_INT, mpiHelper->left_rank,  1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // --- Exchange particle data host <-> host ---
    MPI_Sendrecv(mpiHelper->h_send_left,  left_n       * 6, MPI_FLOAT, mpiHelper->left_rank,  2,
                 mpiHelper->h_recv_right, recv_right_n * 6, MPI_FLOAT, mpiHelper->right_rank, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(mpiHelper->h_send_right, right_n      * 6, MPI_FLOAT, mpiHelper->right_rank, 3,
                 mpiHelper->h_recv_left,  recv_left_n  * 6, MPI_FLOAT, mpiHelper->left_rank,  3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // --- Copy recv buffers host -> device ---
    CHECK(cudaMemcpy(mpiHelper->d_recv_left,  mpiHelper->h_recv_left,
                     recv_left_n  * 6 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mpiHelper->d_recv_right, mpiHelper->h_recv_right,
                     recv_right_n * 6 * sizeof(float), cudaMemcpyHostToDevice));

    // --- Unpack ---
    int recv_left_offset  = keep_n;
    int recv_right_offset = keep_n + recv_left_n;

    if (recv_left_n > 0) {
        unpack_recv_kernel<<<(recv_left_n + threads-1)/threads, block>>>(
            sim->d_new_P, recv_left_offset,
            mpiHelper->d_recv_left, recv_left_n
        );
        CHECK_KERNELCALL();
    }
    if (recv_right_n > 0) {
        unpack_recv_kernel<<<(recv_right_n + threads-1)/threads, block>>>(
            sim->d_new_P, recv_right_offset,
            mpiHelper->d_recv_right, recv_right_n
        );
        CHECK_KERNELCALL();
    }
 
    swap_particles_with_new_another_func(sim);
    sim->NP = keep_n + recv_left_n + recv_right_n;
}
