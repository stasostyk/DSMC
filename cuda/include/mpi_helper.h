#ifndef MPI_HELPER_H
#define MPI_HELPER_H

typedef struct {
    int worldSize;
    int worldRank;
    int numGpus;

    int left_rank;
    int right_rank;   // MPI neighbors (-1 if none)
 
    // MPI Node will only take care of the grid slice with x = [xMin, xMax]
    float slabWidth;
    float xMin;
    float xMax;
    // int kOffset;

    // Three boolean arrays for CUB scans
    // int *d_flag; // now done with d_valid from Simulation sim variable
    int *d_prefix_left, *d_prefix_right;
    int *d_count;

    float *h_send_left;
    float *h_send_right;
    float *h_recv_left;
    float *h_recv_right;

    float *d_send_left;
    float *d_send_right;
    float *d_recv_left_mapped;
    float *d_recv_right_mapped;

    int bufferToSendLeftCount;
    int bufferToSendRightCount;
} MPIHelper;


#endif