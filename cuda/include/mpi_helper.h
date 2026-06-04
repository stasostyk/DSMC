#ifndef MPI_HELPER_H
#define MPI_HELPER_H

#include <mpi.h>

typedef struct {
    int worldSize;
    int worldRank;
    int numGpus;

    int left_rank;
    int right_rank;   // MPI neighbors (-1 if none)
    MPI_Comm comm;

    // MPI Node will only take care of the grid slice with x = [xMin, xMax]
    float slabWidth;
    float xMin;
    float xMax;
    // int kOffset;

    // Three boolean arrays for CUB scans
    int *d_flag;
    int *d_is_left, *d_is_right, *d_is_keep;   // precomputed from d_flag
    int *d_prefix_left, *d_prefix_right, *d_prefix_keep;
    int *d_count;

    float *h_send_left;
    float *h_send_right;
    float *h_recv_left;
    float *h_recv_right;

    float *d_send_left;
    float *d_send_right;
    float *d_recv_left;
    float *d_recv_right;
} MPIHelper;


#endif