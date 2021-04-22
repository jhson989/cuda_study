#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#define cudaErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

float* h_malloc_device (float* h_mem, const int length) {
    float* d_mem;
    cudaErrChk(cudaMalloc ((void **)&d_mem, sizeof(float)*length))
    if (h_mem != nullptr) {
        cudaErrChk(cudaMemcpy (d_mem, h_mem, sizeof(float)*length, cudaMemcpyHostToDevice))
    }
    
    return d_mem;
}


void h_initialize_host(float* A, float* B, float* C, const int length) {
    for (int i=0; i<length; i++) {
        A[i] = i;
        B[i] = i;
        C[i] = 0.0;
    }
}

void h_check_result(const float* A, const float* B, const float* C, const int length) {

    bool success = true;
    for (int i=0; i<length; i++) {
        if (A[i] + B[i] != C[i]) {
            printf("[ERR] result: %f + %f != %f\n", A[i], B[i], C[i]);
            success = false;
        }
    }

    if (success == true)
        printf("Checking results succeed\n");

}


__global__ void vectorAdd (const float *A, const float *B, float *C, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {

    /*** Configuration ***/
    const int length = pow(2, 24);
    struct timeval stime, etime;
    printf("[Vec addiotion of %d elements]\n", length);

    /*** Host memory ***/
    float *h_A = (float*) malloc (length*sizeof(float));
    float *h_B = (float*) malloc (length*sizeof(float));
    float *h_C = (float*) malloc (length*sizeof(float));
    h_initialize_host(h_A, h_B, h_C, length);

    /*** Device memory ***/
    float *d_A=h_malloc_device(h_A, length);;
    float *d_B=h_malloc_device(h_B, length);;
    float *d_C=h_malloc_device(nullptr, length);;


    /*** Launch kernel ***/
    int numThreads = pow(2,10);
    int numBlocks = (length+numThreads-1) / numThreads;
    gettimeofday(&stime, NULL);
    printf("CUDA kernel launched with <<%d, %d>>\n", numBlocks, numThreads);
    vectorAdd<<<numBlocks, numThreads>>> (d_A, d_B, d_C, length);
    cudaDeviceSynchronize();
    gettimeofday(&etime, NULL);
    printf("    Elaped time: %.4f\n", (etime.tv_sec - stime.tv_sec) + ((etime.tv_usec-stime.tv_usec)*10e-6));
    cudaErrChk( cudaGetLastError() );

    /*** Memcpy from device to host ***/
    cudaErrChk( cudaMemcpy(h_C, d_C, sizeof(float)*length, cudaMemcpyDeviceToHost) );
    h_check_result(h_A, h_B, h_C, length);

    /*** Memory free ***/
    cudaDeviceSynchronize();
    cudaErrChk(cudaFree(d_A));
    cudaErrChk(cudaFree(d_B));
    cudaErrChk(cudaFree(d_C));
    free (h_A);
    free (h_B);
    free (h_C);

    return 0;
}
