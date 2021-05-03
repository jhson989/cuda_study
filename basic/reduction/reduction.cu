
#include <cstdio>
#include <cstdlib>

#define DTYPE unsigned long long
#define ull unsigned long long

/*** CUDA API error checking  ***/
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/****************************************************************
  *** Kernel mode : 2
  *** Blocked shared reduction
  ****************************************************************/

/*** Kernel program ***/
__global__ void reduction_blocked_shared (DTYPE* d_data, DTYPE* d_out, ull remain) {
    ull tidx = threadIdx.x;
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ DTYPE smem[];

    if (idx < remain) {
        smem[tidx] = d_data[idx];
    }
    __syncthreads();

    for (ull s=blockDim.x/2; s>0; s>>=1) {
        if (tidx<s && idx+s<remain) {
            smem[tidx]+=smem[tidx+s];
        }
        __syncthreads();
    }
    
    if (tidx == 0) {
        d_out[blockIdx.x] = smem[tidx];
    }
}



/*** Host program ***/
void run_kernel_blocked_shared (DTYPE* d_data, const ull num_data) {

    DTYPE* d_out;
    cudaErrChk (cudaMalloc ((void**)&d_out, sizeof(DTYPE)*num_data));
    ull remain=num_data, next=0;

    dim3 threads (256);
    const size_t size_smem = sizeof (DTYPE) * threads.x;
    while (remain > 1) {
        if (remain%threads.x==0)
            next = remain/threads.x;
        else
            next = remain/threads.x+1;


        dim3 blocks ((remain+threads.x-1)/threads.x);
        reduction_blocked_shared<<<blocks, threads, size_smem>>> (d_data, d_out, remain);
        cudaErrChk (cudaMemcpy (d_data, d_out, next*sizeof(DTYPE), cudaMemcpyDeviceToDevice));
        cudaErrChk (cudaDeviceSynchronize ())
        cudaErrChk (cudaGetLastError() );
        
        remain = next;
    } 

    cudaErrChk (cudaFree (d_out));
 
}




/****************************************************************
  *** Kernel mode : 1
  *** Blocked reduction
  ****************************************************************/

/*** Kernel program ***/
__global__ void reduction_blocked (DTYPE* d_data, DTYPE* d_out, ull remain) {
    ull tidx = threadIdx.x;
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (ull s=blockDim.x/2; s>0; s>>=1) {
        if (tidx<s && idx+s < remain) {
            d_data[idx]+=d_data[idx+s];
        }
        __syncthreads();
    }
    
    if (tidx == 0) {
        d_out[blockIdx.x] = d_data[idx];
    }
}


/*** Host program ***/
void run_kernel_blocked (DTYPE* d_data, const ull num_data) {

    DTYPE* d_out;
    cudaErrChk (cudaMalloc ((void**)&d_out, sizeof(DTYPE)*num_data));
    ull remain=num_data, next=0;

    dim3 threads (256);
    while (remain > 1) {
        if (remain%threads.x==0)
            next = remain/threads.x;
        else
            next = remain/threads.x+1;


        dim3 blocks ((remain+threads.x-1)/threads.x);
        reduction_blocked<<<blocks, threads>>> (d_data, d_out, remain);
        cudaErrChk (cudaMemcpy (d_data, d_out, next*sizeof(DTYPE), cudaMemcpyDeviceToDevice));
        cudaErrChk (cudaDeviceSynchronize ())
        cudaErrChk (cudaGetLastError() );
        
        remain = next;
    } 

    cudaErrChk (cudaFree (d_out));
 
}




/****************************************************************
  *** Kernel mode : 0
  *** Basic reduction
  ****************************************************************/

/*** Kernel program ***/
__global__ void reduction (DTYPE* d_data, ull remain, ull next) {
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx + next < remain) {
        d_data[idx] += d_data[idx+next];
    }
}


/*** Host program ***/
void run_kernel_basic (DTYPE* d_data, const ull num_data) {

    ull remain=num_data, next=0;
    while (remain > 1) {
        if (remain%2==0)
            next = remain/2;
        else
            next = remain/2 +1;

        dim3 threads (256);
        dim3 blocks ((next+threads.x-1)/threads.x);
        reduction<<<blocks, threads>>> (d_data, remain, next);
        cudaErrChk (cudaDeviceSynchronize ())
        cudaErrChk (cudaGetLastError() );
        remain = next;
    }
}





/****************************************************************
  *** Host program
  ****************************************************************/

DTYPE initial_data (DTYPE* data, const ull num_data) {
    DTYPE sum = 0;
    for (ull i=0; i<num_data; i++) {
        data[i] = rand()%5;
        sum += data[i];
    }
    return sum;
}

int select_mode(const int argc, const char** argv) {

    int mode = 0;
    if (argc > 1)
        mode = atoi(argv[1]);

    switch (mode) {
        case 0:
            printf("Kernel mode : 0.Basic reduction\n");
            break;
        case 1:
            printf("Kernel mode : 1.Blocked reduction\n");
            break;
        case 2:
            printf("Kernel mode : 2.Blocked shared reduction\n");
            break;
        default:
            printf("Selected not implemented mode...\n");
            exit(1);
            break;
    }
    return mode;
}


int main (const int argc, const char** argv) {

    /*** Program Configuration ***/
    const ull num_data = 4*1e+8;
    const int loop_exe = 4;
    const size_t size_data = sizeof (ull) * num_data;

    printf("\n\n=======================================================================\n");
    printf("== Parallel DTYPE reduction\n");
    printf("=======================================================================\n");
    const int mode_kernel = select_mode(argc, argv);
    printf("Number of DTYPE : %llu\n", num_data);
    printf("    size of mem : %.2f GB\n", size_data*1e-9);


    /*** Initialize variables ***/
    DTYPE* data = (DTYPE*) malloc (size_data);
    const DTYPE sum = initial_data (data, num_data);
    float gops = 1e-9*num_data*loop_exe;
    cudaEvent_t start, stop;
    float msec_total=0.0f, msec=0.0f;
    cudaErrChk (cudaEventCreate(&start));
    cudaErrChk (cudaEventCreate(&stop));


    /*** Set CUDA Memory ***/
    DTYPE* d_data;
    cudaErrChk (cudaMalloc ((void**)&d_data, size_data));
    cudaErrChk (cudaMemcpy (d_data, data, size_data, cudaMemcpyHostToDevice));
    cudaErrChk (cudaDeviceSynchronize ());


    /*** Run kernel ***/
    for (int loop=0; loop<loop_exe; loop++) {
        cudaErrChk (cudaMemcpy (d_data, data, size_data, cudaMemcpyHostToDevice));
        cudaErrChk (cudaEventRecord(start, NULL));
        switch (mode_kernel) {
            case 0:
                run_kernel_basic (d_data, num_data);
                break;
            case 1:
                run_kernel_blocked (d_data, num_data);
                break;
            case 2:
                run_kernel_blocked_shared (d_data, num_data);
                break;
            default:
                printf("Not implemented\n");
                exit(1);
                break;
        }
        cudaErrChk (cudaEventRecord(stop, NULL));
        cudaErrChk (cudaEventSynchronize(stop));
        cudaErrChk (cudaEventElapsedTime(&msec, start, stop));
        msec_total += msec;
    }

    /*** Check result ***/
    DTYPE result;
    cudaErrChk (cudaMemcpy (&result, d_data, sizeof (DTYPE), cudaMemcpyDeviceToHost));

    printf("    Total number of floating point multiplications : %.2f Gops\n", gops);
    printf("    Elaped time: %.4f msec\n", msec_total);
    printf("    GFLOPS : %.4f gflops [Avg. of %d time(s)]\n", gops/(msec_total*1e-3), loop_exe); 


    printf("Check result ...\n");
    if (sum != result) {
        printf("    [Err] GT(%llu) != Pred(%llu)\n", sum, result);
    } else {
        printf("    [Pass] GT(%llu) == Pred(%llu)\n", sum, result);
    }
    printf("=======================================================================\n\n");

    /*** Finalize program ***/
    cudaErrChk (cudaFree (d_data));    
    free (data);

    return 0;
}
