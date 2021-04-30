
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



/********************************
  *** Kernel program
  ********************************/

__global__ void reduction (DTYPE* d_data, ull remain, ull next) {
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx + next < remain) {
        d_data[idx] += d_data[idx+next];
    }
}







/********************************
  *** Host program
  ********************************/


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

    /*** Run kernel ***/
    ull remain=num_data, next=0;
    while (remain > 1e+3) {
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



    /*** Check result ***/
    DTYPE* result = (DTYPE*) malloc (sizeof (DTYPE)*remain);
    cudaErrChk (cudaMemcpy (result, d_data, sizeof (DTYPE)*remain, cudaMemcpyDeviceToHost));
    for (int i=1; i<remain; i++)
        result[0] += result[i];


    printf("Check result ...\n");
    if (sum != result[0]) {
        printf("Err GT(%llu) != Pred(%llu)\n", sum, result[0]);
    } else {
        printf("Pass GT(%llu) == Pred(%llu)\n", sum, result[0]);
    }
    printf("=======================================================================\n\n");

    /*** Finalize program ***/
    cudaErrChk (cudaFree (d_data));    
    free (data);
    free (result);

    return 0;
}
