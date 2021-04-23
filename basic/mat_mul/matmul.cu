
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/time.h>
#include "kernel.hpp"

#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void matmul_serial (const float *A, const float *B, float *C, const int len) {
    printf("[CPU] Kernel start..\n");
    

    timeval st, ed;
    gettimeofday(&st, NULL);
    // Main body
    for (int i=0; i<len; i++) {
        for (int j=0; j<len; j++) {
            float sum = 0;
            for (int k=0; k<len; k++) {
                sum += A[i*len+k]*B[k*len+j];
            }
            C[i*len+j] = sum;
        }
    }
    // End of main body
    gettimeofday(&ed, NULL);

    float time = (ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6);
    float gops = 1.0*len*len*len*1e-9;
    printf("    Total number of floating point multiplications : %.2fGops\n", gops);
    printf("    Elaped time: %.4f\n", time);
    printf("    GFLOPS : %.4f\n", gops/time); 


}


void matmul_cuda_basic (const float *A, const float *B, float *C, const int len) {

    /***
      CUDA implementataion without any optimization methods
      **/
    const int num_threads = 32;
    const dim3 dim_threads(num_threads, num_threads);
    const dim3 dim_blocks((len+num_threads-1)/num_threads, (len+num_threads-1)/num_threads);
    printf("[GPU] Basic kernel start..\n");
    printf("    Grid size: [(%d, %d), (%d, %d)]\n", dim_blocks.y, dim_blocks.x, dim_threads.y, dim_threads.x);


    /*** Memcpy H to D ***/
    float *d_A, *d_B, *d_C;
    cudaErrChk (cudaMalloc ((void **)&d_A, sizeof(float)*len*len));
    cudaErrChk (cudaMalloc ((void **)&d_B, sizeof(float)*len*len));
    cudaErrChk (cudaMalloc ((void **)&d_C, sizeof(float)*len*len));
    cudaErrChk (cudaMemcpy (d_A, A, sizeof(float)*len*len, cudaMemcpyHostToDevice));
    cudaErrChk (cudaMemcpy (d_B, B, sizeof(float)*len*len, cudaMemcpyHostToDevice));
    

    timeval st, ed;
    gettimeofday(&st, NULL);
    // Main body
    matmul_basic<<<dim_blocks, dim_threads>>>(d_A, d_B, d_C, len);
    cudaErrChk (cudaDeviceSynchronize ())
    cudaErrChk( cudaGetLastError() );
    // End of main body
    gettimeofday(&ed, NULL);

    float time = (ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6);
    float gops = 1.0*len*len*len*1e-9;
    printf("    Total number of floating point multiplications : %.2fGops\n", gops);
    printf("    Elaped time: %.4f\n", time);
    printf("    GFLOPS : %.4f\n", gops/time); 

    cudaErrChk (cudaMemcpy(C, d_C, sizeof(float)*len*len, cudaMemcpyDeviceToHost));
    cudaErrChk (cudaDeviceSynchronize ())
    cudaErrChk (cudaFree (d_A));
    cudaErrChk (cudaFree (d_B));
    cudaErrChk (cudaFree (d_C));

}



void matmul_cuda_shared (const float *A, const float *B, float *C, const int len) {

    /***
      CUDA implementataion without any optimization methods
      **/
    const int len_tile = 32;
    const dim3 dim_threads(len_tile, len_tile);
    const dim3 dim_blocks((len+len_tile-1)/len_tile, (len+len_tile-1)/len_tile);
    const int size_smem = 2*sizeof(float)*len_tile*len_tile;
    printf("[GPU] Kernel with shared memory start..\n");
    printf("    Grid size: [(%d, %d), (%d, %d)]\n", dim_blocks.y, dim_blocks.x, dim_threads.y, dim_threads.x);
    printf("    Shared mem size: %.2fKB\n", size_smem/1024.0);


    /*** Memcpy H to D ***/
    float *d_A, *d_B, *d_C;
    cudaErrChk (cudaMalloc ((void **)&d_A, sizeof(float)*len*len));
    cudaErrChk (cudaMalloc ((void **)&d_B, sizeof(float)*len*len));
    cudaErrChk (cudaMalloc ((void **)&d_C, sizeof(float)*len*len));
    cudaErrChk (cudaMemcpy (d_A, A, sizeof(float)*len*len, cudaMemcpyHostToDevice));
    cudaErrChk (cudaMemcpy (d_B, B, sizeof(float)*len*len, cudaMemcpyHostToDevice));
    

    timeval st, ed;
    gettimeofday(&st, NULL);
    // Main body
    matmul_tiled<<<dim_blocks, dim_threads, size_smem>>>(d_A, d_B, d_C, len, len_tile);
    cudaErrChk (cudaDeviceSynchronize ())
    cudaErrChk( cudaGetLastError() );
    // End of main body
    gettimeofday(&ed, NULL);

    float time = (ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6);
    float gops = 1.0*len*len*len*1e-9;
    printf("    Total number of floating point multiplications : %.2fGops\n", gops);
    printf("    Elaped time: %.4f\n", time);
    printf("    GFLOPS : %.4f\n", gops/time); 

    cudaErrChk (cudaMemcpy(C, d_C, sizeof(float)*len*len, cudaMemcpyDeviceToHost));
    cudaErrChk (cudaDeviceSynchronize ())
    cudaErrChk (cudaFree (d_A));
    cudaErrChk (cudaFree (d_B));
    cudaErrChk (cudaFree (d_C));

}



/****************************************
  ************** Host Code **************
  ****************************************/

void h_initialize(float *mem, const int len) {
    for (int i=0; i<len; i++) {
        for (int j=0; j<len; j++) {
            mem[i*len+j] = (float)(rand()%1000);
        }
    }
}



bool h_test(const float *A, const float *B, const float *C, const int len) {

    printf("[TEST] Test start..\n");

    for (int i=0; i<len; i++) {
        for (int j=0; j<len; j++) {
            float sum = 0;
            for (int k=0; k<len; k++) {
                sum += A[i*len+k]*B[k*len+j];
            }
            if (sum != C[i*len+j]){
                printf("C[%d][%d] = %.f != %f\n", i, j, C[i*len+j], sum);
                return false;
            }
        }
    }
    return true;
}


int main(int argc, char** argv) {

    /*** Program configuration ***/
    printf("\n============================================\n");
    printf("Matrix multiplication\n");
    printf("    A * B = C\n");
    printf("    arg : ./matmul [len] [Test:0,1]\n");
    printf("============================================\n\n");
    int len = (int)1e+3;
    if (argc >= 2) 
        len = atoi(argv[1]);
    srand(0);

    /*** Data initialize ***/
    float *A = (float *) malloc (len*len*sizeof(float));
    float *B = (float *) malloc (len*len*sizeof(float));
    float *C = (float *) calloc (len*len,sizeof(float));
    h_initialize(A, len);
    h_initialize(B, len);
    printf("[Mem] Size of a matrix : [%d, %d]\n", len, len);
    printf("[Mem] Total size of matrices : %.3fGB\n", 3.0*len*len*sizeof(float)*1e-9);


    /*** Run a matmul ***/
    //matmul_serial (A, B, C, len);
    matmul_cuda_basic (A, B, C, len);
    matmul_cuda_shared (A, B, C, len);

    /*** Test the result ***/
    if (argc == 3 && atoi(argv[2]) == 0) {
        printf("[TEST] Test skipped..\n");
    } else {
        if (h_test (A, B, C, len) == true) {
            printf("    Test passed\n");
        } else {
            printf("    [ERR] Test failed!!\n");
        }
    }
    /*** Finalize ***/
    free (A);
    free (B);
    free (C);


    printf("============================================\n\n");
    return 0;
}


