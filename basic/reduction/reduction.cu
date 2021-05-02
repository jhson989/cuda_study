
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
  *** Blocked reduction
  ****************************************************************/

/*** Kernel program ***/
__global__ void reduction_blocked (DTYPE* d_data, DTYPE* d_out, ull remain) {
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;
    ull 
    DTYPE sum = 0;

    for (int s=blockDim.x>>1; s>0; s>>=1) {
        if (idx + s < remain) {
            d_data[out]+=d_data[out_s];
        }
        __syncthreads();
    }
    
    d_out[blockIdx.x] = sum;
}


/*** Host program ***/
ull run_kernel_basic (DTYPE* d_data, const ull num_data) {

    DTYPE* d_out;
    cudaErrChk (cudaMalloc ((void**)&d_out, sizeof(DTYPE)*num_data));
    ull remain=num_data, next=0;

    dim3 threads (256);
    while (remain > 1) {
        next = remain/thread.x + remain%thread.x;

        dim3 blocks ((remain+threads.x-1)/threads.x);
        reduction_blocked<<<blocks, threads>>> (d_data, d_out, remain, next);
        cudaErrChk (cudaMemcpy (d_data, d_out, next*sizeof(DYPT), cudaMemcpyDeviceToDevice));
        cudaErrChk (cudaDeviceSynchronize ())
        cudaErrChk (cudaGetLastError() );
        remain = next;
    }


    cudaErrChk (cudaFree (d_out));
    return remain;
}




/****************************************************************
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
ull run_kernel_basic (DTYPE* d_data, const ull num_data) {

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

    return remain;
}





/****************************************************************
  *** Host program
  ****************************************************************/

DTYPE initial_data (DTYPE* data, const ull num_data) {
    DTYPE sum = 0;
    for (ull i=0; i<num_data; i++) {
        data[i] = rand()%5-2;
        sum += data[i];
    }
    return sum;
}


int main (int argc, char** argv) {

    /*** Program Configuration ***/
    const ull num_data = 5*1e+8;
    const size_t size_data = sizeof (ull) * num_data;
    printf("\n\n=======================================================================\n");
    printf("== Parallel integer reduction\n");
    printf("=======================================================================\n");
    printf("Number of int : %llu\n", num_data);
    printf("    size of mem : %.2f GB\n", size_data*1e-9);


    /*** Initialize variables ***/
    DTYPE* data = (DTYPE*) malloc (size_data);
    const DTYPE sum = initial_data (data, num_data);



    /*** Set CUDA Memory ***/
    DTYPE* d_data;
    cudaErrChk (cudaMalloc ((void**)&d_data, size_data));
    cudaErrChk (cudaMemcpy (d_data, data, size_data, cudaMemcpyHostToDevice));
    cudaErrChk (cudaDeviceSynchronize ())

    /*** Run kernel ***/
    run_kernel_basic (d_data, num_data);

    /*** Check result ***/
    DTYPE result;
    cudaErrChk (cudaMemcpy (&result, d_data, sizeof (DTYPE), cudaMemcpyDeviceToHost));
    
    printf("Check result ...\n");
    if (sum != result[0]) {
        printf("Err GT(%llu) != Pred(%llu)\n", sum, result);
    } else {
        printf("Pass GT(%llu) == Pred(%llu)\n", sum, result);
    }
    printf("=======================================================================\n\n");

    /*** Finalize program ***/
    cudaErrChk (cudaFree (d_data));    
    free (data);

    return 0;
}
