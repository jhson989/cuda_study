
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/time.h>

#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct config {
    int AH;
    int AW;
    int BH;
    int BW;
    int CH;
    int CW;
    float alpha;
    float beta;
    bool do_test;
};


/***************************************
  * Device code "matmul"
  **************************************/

template <int BLOCK_SIZE> 
__global__ void sgemm (const float* A, const float* B, float* C, const int CH, const int CW, const int AW, const int BW, const float alpha, const float beta) {

    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int sy = threadIdx.y;
    int sx = threadIdx.x;

    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    int num_tiles = (AW+(BLOCK_SIZE-1))/BLOCK_SIZE;

    float sum=0.0f;
    for (int t=0; t<num_tiles; t++) {
        if (y<CH && (sx+t*BLOCK_SIZE)<AW) {
            sA[sy][sx] = A[y*AW+(sx+t*BLOCK_SIZE)];
        } else {
            sA[sy][sx] = 0.0f;
        }

        if (x<CW && (sy+t*BLOCK_SIZE)<AW) {
            sB[sy][sx] = B[(sy+t*BLOCK_SIZE)*BW+x];
        } else {
            sB[sy][sx] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k=0; k<BLOCK_SIZE; k++) {
            sum += sA[sy][k]*sB[k][sx];
        }
        __syncthreads();
    }

    if (x<CW && y<CH) {
        C[y*CW+x] = alpha*sum + beta*C[y*CW+x];
    }

}


/***************************************
  * Host code "matmul"
  **************************************/
float* host_mat_mul(const float* A, const float* B, const float* C, const struct config conf) {

    printf("[Kernel] Run kernal\n");
    /*** Initialize device memory ***/
    const int num_executions = 1;
    const int block_size = 16;
    size_t size_A = sizeof(float)*conf.AH*conf.AW;
    size_t size_B = sizeof(float)*conf.BH*conf.BW;
    size_t size_C = sizeof(float)*conf.CH*conf.CW;
    float *d_A, *d_B, *d_C;
    float *result = (float *) malloc (conf.CH*conf.CW*sizeof(float));
    cudaErrChk (cudaMalloc ((void**)(&d_A), size_A));
    cudaErrChk (cudaMalloc ((void**)(&d_B), size_B));
    cudaErrChk (cudaMalloc ((void**)(&d_C), size_C));
    cudaErrChk (cudaMemcpy (d_A, A, size_A, cudaMemcpyHostToDevice));
    cudaErrChk (cudaMemcpy (d_B, B, size_B, cudaMemcpyHostToDevice));
    cudaErrChk (cudaMemcpy (d_C, C, size_C, cudaMemcpyHostToDevice));
    cudaErrChk (cudaDeviceSynchronize ())

    /*** Setup execution config ***/
    dim3 threads(block_size, block_size);
    dim3 blocks((conf.CW+threads.x-1)/threads.x, (conf.CH+threads.y-1)/threads.y);

    /*** Run CUDA kernel ***/
    cudaEvent_t start, stop;
    cudaErrChk(cudaEventCreate(&start));
    cudaErrChk(cudaEventCreate(&stop));
    cudaErrChk(cudaEventRecord(start, NULL));
    // Main body
    for (int i=0; i<num_executions; i++) {
        sgemm<block_size><<<blocks, threads>>>(d_A, d_B, d_C, conf.CH, conf.CW, conf.AW, conf.BW, conf.alpha, conf.beta);
        cudaErrChk (cudaDeviceSynchronize ())
        cudaErrChk( cudaGetLastError() );
    }
    // End of main body
    cudaErrChk(cudaEventRecord(stop, NULL));
    cudaErrChk(cudaEventSynchronize(stop));
    float msec_total = 0.0f;
    float gflo = num_executions*conf.CH*conf.CW*(2.0*conf.AW+2)*1e-9;
    cudaErrChk(cudaEventElapsedTime(&msec_total, start, stop));
    printf("    Elaped time: %.4f msec\n", msec_total);
    printf("    gFlops : %.4f gflops\n", gflo/(msec_total*1e-3));

    cudaErrChk (cudaMemcpy(result, d_C, size_C, cudaMemcpyDeviceToHost));
    cudaErrChk (cudaDeviceSynchronize ())
    cudaErrChk (cudaFree (d_A));
    cudaErrChk (cudaFree (d_B));
    cudaErrChk (cudaFree (d_C));

    return result;
}



/****************************************
  * Helper functions for host
  ****************************************/

const struct config host_get_cmd_args(int argc, char** argv) {

    int a=100, b=100, c=100;
    float alpha=1.0f, beta=0.0f;
    bool do_test = false;

    if (argc >= 2)
        do_test = (bool)atoi(argv[1]);
    if (argc >= 7) {
        a = atoi(argv[2]);
        b = atoi(argv[3]);
        c = atoi(argv[4]);
        alpha = atof(argv[5]);
        beta = atof(argv[6]);
    }

    struct config conf = {
        a,
        b,
        b,
        c,
        a,
        c,
        alpha,
        beta,
        do_test
    };
    printf("\n================================================\n");
    printf("CUDA implementaion of SGEMM\n");
    printf("    args: ./matmul [test] [a, b, c, alpha, beta]\n");
    printf("    C[a, c] = alpha * A[a, b] @ B[b, c] + beta * C[a, c]\n");
    printf("    C[%d, %d] = %f * A[%d, %d] @ B[%d, %d] + %f * C[%d, %d]\n", a,c,alpha,a,b,b,c,beta,a,c);
    printf("================================================\n\n");
    return conf;
}


void host_initialize(float *mem, const int H, const int W) {
    for (int i=0; i<H; i++) {
        for (int j=0; j<W; j++) {
            mem[i*W+j] = (float)(rand()%100);
        }
    }
}

void host_test(const float *A, const float *B, const float *C, const float * result, const struct config conf) {

    if (conf.do_test == false) {
        printf("[TEST] Test skipped..\n");
        return;
    }

    printf("[TEST] Test start..\n");

    float alpha=conf.alpha, beta=conf.beta;
    int len_k = conf.AW;
    for (int i=0; i<conf.CH; i++) {
        for (int j=0; j<conf.CW; j++) {
            float sum = 0;
            for (int k=0; k<len_k; k++) {
                sum += A[i*conf.AW+k]*B[k*conf.BW+j];
            }
            sum = alpha*sum+beta*C[i*conf.CW+j];
            if (sum != result[i*conf.CW+j]){
                printf("    [ERROR] C[%d][%d] = %.f != %f\n", i, j, result[i*conf.CW+j], sum);
                printf("    Test failed...!\n");
                return;
            }
        }
    }
    printf("    Test passed!!\n");
    return;
}


/***************************************
  * Main function
  **************************************/
int main(int argc, char** argv) {

    /*** Program configuration ***/
    const struct config conf = host_get_cmd_args(argc, argv);
    srand(0);

    /*** Initialize Data ***/
    float *A = (float *) malloc (conf.AH*conf.AW*sizeof(float));
    float *B = (float *) malloc (conf.BH*conf.BW*sizeof(float));
    float *C = (float *) calloc (conf.CH*conf.CW,sizeof(float));
    host_initialize(A, conf.AH, conf.AW);
    host_initialize(B, conf.BH, conf.BW);
    host_initialize(C, conf.CH, conf.CW);
    size_t total_size = (size_t)(conf.AH*conf.AW*sizeof(float) + conf.BH*conf.BW*sizeof(float) + 2.0*conf.CH*conf.CW*sizeof(float));
    printf("[Mem] Total size of matrices : %.3fGB\n", total_size*1e-9);

    /*** Run matmul ***/
    float* result = host_mat_mul (A, B, C, conf); 

    /*** Test result ***/
    host_test(A, B, C, result, conf);

    /*** Finalize ***/
    free (A);
    free (B);
    free (C);
    free (result);

    return 0;
}


