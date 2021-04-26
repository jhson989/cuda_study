
#ifndef __KERNEL__
#define __KERNEL__

__global__ void matmul_basic(const float* A, const float* B, float* C, const int len);

__global__ void matmul_tiled(const float* A, const float* B, float* C, const int len, const int tile_size);

__global__ void transpose(const float* A, float* A_T, const int len);

__global__ void matmul_tiled_transposed(const float* A, const float* B, float* C, const int len, const int len_tile);
#endif

