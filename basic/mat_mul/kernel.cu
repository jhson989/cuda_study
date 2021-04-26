
#include <cstdio>
#include "kernel.cuh"


__global__ void matmul_basic(const float* A, const float* B, float* C, const int len) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<len&&j<len) {
        float sum=0;
        for (int k=0; k<len; k++) {
            sum += A[i*len+k]*B[k*len+j];
        }
        C[i*len+j]=sum;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, const int len, const int len_tile) {

    extern __shared__ float smem[];
    float *sA = &smem[0];
    float *sB = &smem[len_tile*len_tile];


    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int si = threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int sj = threadIdx.x;
    int num_tiles = (len+(len_tile-1))/len_tile;

    float sum = 0;
    for (int t=0; t<num_tiles; t++) {
        if (i<len && (sj+t*len_tile) < len)
            sA[si*len_tile+sj] = A[(i)*len+(sj+t*len_tile)];
        else
            sA[si*len_tile+sj] = 0;

        if (j<len && (si+t*len_tile) < len)
            sB[si*len_tile+sj] = B[(si+t*len_tile)*len+(j)];
        else 
            sB[si*len_tile+sj] = 0;
        __syncthreads();

        for (int k=0; k<len_tile; k++) {
            sum += sA[(si*len_tile)+k]*sB[(k*len_tile)+sj];
        }
        __syncthreads();
    }
    if (i<len && j<len) {
        C[i*len+j] = sum;
    }

}


__global__ void transpose(const float* A, float* A_T, const int len) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=j && i<len) {
        A_T[i*len+j] = A[j*len+i];
        A_T[j*len+i] = A[i*len+j];
    }
}

__global__ void matmul_tiled_transposed(const float* A, const float* B, float* C, const int len, const int len_tile) {

    extern __shared__ float smem[];
    float *sA = &smem[0];
    float *sB = &smem[len_tile*len_tile];


    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int si = threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int sj = threadIdx.x;
    int num_tiles = (len+(len_tile-1))/len_tile;
    int itosj = ((i/len_tile)*len_tile) + sj;

    float sum = 0;
    for (int t=0; t<num_tiles; t++) {
        if (itosj<len && (si+t*len_tile) < len)
            sA[si*len_tile+sj] = A[(si+t*len_tile)*len+(itosj)];
        else
            sA[si*len_tile+sj] = 0;

        if (j<len && (si+t*len_tile) < len)
            sB[si*len_tile+sj] = B[(si+t*len_tile)*len+(j)];
        else 
            sB[si*len_tile+sj] = 0;
        __syncthreads();

        for (int k=0; k<len_tile; k++) {
            sum += sA[(k*len_tile)+si]*sB[(k*len_tile)+sj];
        }
        __syncthreads();
    }
    if (i<len && j<len) {
        C[i*len+j] = sum;
    }
}




