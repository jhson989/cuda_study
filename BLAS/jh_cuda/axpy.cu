
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/time.h>


/*******************************************************
 ****************** Device code ************************
 ******************************************************/

__constant__ double d_alpha;

__global__ void axpy (const double* A, const double* B, double* C, const unsigned int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements)
        C[idx] = d_alpha*A[idx]+B[idx];

}




/*******************************************************
 ******************** Host code ************************
 ******************************************************/

#define cudaErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void h_init_value(double* mem, const unsigned int num_elements) {
    for (int i=0; i<num_elements; i++)
        mem[i] = (double) rand();
}

void h_test (const double* A, const double* B, const double* C, const double alpha, const unsigned int num_elements) {
    bool success = true;

    for (int i=0; i<num_elements; i++) {
        if (alpha*A[i]+B[i] != C[i])
            success = false;
    }
    
    if (success) {
        printf("[TEST] Test passed\n");
    } else {
        printf("[TEST] Test failed\n");
    }
}



int main(int argc, char** argv) {
    printf("\n");
    printf("=========================================================================\n");
    printf("[BLAS} axpy implementation \n");
    printf("=========================================================================\n");
    printf("\n");
    
    /*** Configuration ***/
    unsigned int num_elements = 3e+8;
    size_t size = num_elements*sizeof(double);
    timeval stime, etime;
    double alpha = 1.0;
    if (argc == 2)
        alpha = atof(argv[1]);


    /*** Mem allocation ***/
    double *h_A=nullptr, *h_B=nullptr, *h_C=nullptr; // host(CPU)-side variable
    double *d_A=nullptr, *d_B=nullptr, *d_C=nullptr; // device(GPU)-side variable
    h_A = (double*) malloc (size);
    h_B = (double*) malloc (size);
    h_C = (double*) malloc (size);
    cudaErrChk ( cudaMalloc ((void**)&d_A, size) );
    cudaErrChk ( cudaMalloc ((void**)&d_B, size) );
    cudaErrChk ( cudaMalloc ((void**)&d_C, size) );
    cudaErrChk ( cudaMemcpyToSymbol (d_alpha, &alpha, sizeof(double)) );
    printf("[mem] Allocated : 3 doulbe precision vectors[%u-D]. %.2fGB for each devices [CPU, GPU]\n"
                , num_elements, 3*(double)size/1024/1024/1024);


    /*** Program init ***/
    h_init_value (h_A, num_elements);
    h_init_value (h_B, num_elements);
    cudaErrChk ( cudaMemcpy (d_A, h_A, size, cudaMemcpyHostToDevice) );
    cudaErrChk ( cudaMemcpy (d_B, h_B, size, cudaMemcpyHostToDevice) );


    /*** Launch a kernel ***/
    unsigned int num_threads = 1024;
    unsigned int num_blocks = (num_elements + (num_threads-1))/num_threads;

    printf("[kernel] <%u, %u>-size grid launched\n"
                , num_blocks, num_threads);
    gettimeofday(&stime, NULL);
    axpy<<<num_blocks, num_threads>>>(d_A, d_B, d_C, num_elements);
    cudaErrChk ( cudaDeviceSynchronize () )
    cudaErrChk ( cudaGetLastError () ); 
    gettimeofday(&etime, NULL);
    printf("[kernel] Elapsed time: %.4f\n" 
                , ((etime.tv_sec-stime.tv_sec)+(etime.tv_usec-stime.tv_usec)*10e-6) );


    /*** Test computed result ***/
    cudaErrChk ( cudaMemcpy (h_C, d_C, size, cudaMemcpyDeviceToHost) );
    h_test (h_A, h_B, h_C, alpha, num_elements);


    /*** Finalize ***/
    free (h_A);
    free (h_B);
    free (h_C);
    cudaErrChk (cudaFree (d_A));
    cudaErrChk (cudaFree (d_B));
    cudaErrChk (cudaFree (d_C));

    return 0;

}






