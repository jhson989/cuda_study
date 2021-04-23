#include <cstdio>

#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int main(void) {
    
    int num_devices=0;
    cudaErrChk ( cudaGetDeviceCount (&num_devices) );
    printf("\n=================================================\n");
    printf("The number of device(s) : %d\n", num_devices);
    printf("=================================================\n\n");

    for (int i=0; i<num_devices; i++) {
        
        cudaDeviceProp prop;
        cudaErrChk ( cudaGetDeviceProperties (&prop, i) );

        printf ("Device Number: %d\n", i);
        printf ("  Device name: %s\n", prop.name);
        printf ("  Device compute capability: %d.%d\n", prop.major, prop.minor);
        printf ("  Number of SM(s): %d\n", prop.multiProcessorCount);
        printf ("  Memory Clock Rate (GHz): %.2f\n",
               ((float)prop.memoryClockRate)/1.0e6);
        printf ("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf ("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

        printf ("\n[Kernel size]\n");
        printf ("  Maximum size of a grid [%d, %d, %d]\n"
                , prop.maxGridSize[0], prop.maxGridSize[0], prop.maxGridSize[0]);
        printf ("  Maximum size of a block [%d]\n"
                , prop.maxThreadsPerBlock);
        printf ("\n[Shared mem]\n");
        printf ("  Shared memory size per block :%dKB\n", (int)(prop.sharedMemPerBlock/1.0e3));

    }

    printf("\n=================================================\n\n");
    return 0;
}

