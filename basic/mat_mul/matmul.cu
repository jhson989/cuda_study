
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/time.h>

void matmul_serial(const float *A, const float *B, float *C, const int len) {


    printf("[CPU] Kernel Start..\n");
    float gops = 1.0*len*len*len*1e-9;
    printf("    Total number of floating point multiplications : %.2fGops\n", gops);

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
    printf("    Elaped time: %.4f\n", time);
    printf("    GFLOPS : %.4f\n", gops/time); 


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
            if (sum != C[i*len+j])
                return false;
        }
    }
    return true;
}


int main(int argc, char** argv) {

    /*** Program configuration ***/
    printf("\n============================================\n");
    printf("Matrix multiplication\n");
    printf("A * B = C\n");
    printf("============================================\n\n");
    int len = (int)1e+3;
    if (argc == 2) 
        len = atoi(argv[1]);

    /*** Data initialize ***/
    float *A = (float *) malloc (len*len*sizeof(float));
    float *B = (float *) malloc (len*len*sizeof(float));
    float *C = (float *) calloc (len*len,sizeof(float));
    h_initialize(A, len);
    h_initialize(B, len);
    printf("[Mem] Size of a matrix : [%d, %d]\n", len, len);
    printf("[Mem] Total size of matrices : %.3fGB\n", 3.0*len*len*sizeof(float)*1e-9);


    /*** Run a matmul ***/
    matmul_serial (A, B, C, len);

    /*** Test the result ***/
    if (h_test (A, B, C, len) == true) {
        printf("Test passed\n");
    } else {
        printf("[ERR] Test failed!!\n");
    }

    /*** Finalize ***/
    free (A);
    free (B);
    free (C);


    printf("============================================\n\n");
    return 0;
}


