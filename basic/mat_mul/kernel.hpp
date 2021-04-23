
#ifndef __KERNEL__
#define __KERNEL__

__global__ void matmul_basic(const float* A, const float* B, float* C, const int len);

__global__ void matmul_tiled(const float* A, const float* B, float* C, const int len, const int tile_size);
#endif

