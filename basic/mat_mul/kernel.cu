

__global__ void matmul_basic(const float* A, const float* B, float* C, const int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i=idx/len, j=idx%len;
    float sum=0;
    for (int k=0; k<len; i++) {
        sum += A[i*len+k]*B[k*len+j];
    }
    C[idx]=sum;
}

