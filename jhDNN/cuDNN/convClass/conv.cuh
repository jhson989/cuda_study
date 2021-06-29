#ifndef __JH_CONV__
#define __JH_CONV__

#include <cstdio>
#include <cstdlib>
#include <cudnn.h>

#define cudnnErrChk(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUDNN_STATUS_SUCCESS) 
    {
         fprintf(stderr,"cuDNN assert: %s %s %d\n", cudnnGetErrorString(code), file, line);
         if (abort) exit(code);
    }
}

#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
   }
}


class Conv2d {

private:
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t desc_input;
    cudnnTensorDescriptor_t desc_output;
    cudnnFilterDescriptor_t desc_kernel;

    cudnnConvolutionDescriptor_t desc_layer;
    cudnnConvolutionFwdAlgo_t conv_algo;
    size_t size_workspace;
    float* d_workspace = nullptr;
    float * d_kernel = nullptr;
    int BATCH_SIZE, O_C, O_H, O_W;

public:
    Conv2d(
        /*BATCH_SIZE*/int N,
        /*INPUT_CHANNEL*/int IN_C,
        /*OUTPUT_CHANNEL*/int OUT_C,
        /*IMAGE_HEIGHT*/int IMG_H,
        /*IMAGE_WIDTH*/int IMG_W,
        /*KERNEL_HEIGHT*/int K_H,
        /*KERNEL_WIDTH*/int K_W,
        /*PAD_H*/int PAD_H=1,
        /*PAD_W*/int PAD_W=1,
        /*STRIDE_H*/int STRIDE_H=1,
        /*STRIDE_W*/int STRIDE_W=1,
        /*DIALTION_H*/int DIALTION_H=1,
        /*DIALTION_W*/int DIALTION_W=1
    );
    
    float* run(const float* d_input);
    ~Conv2d();

    inline size_t get_size_workspace() {return size_workspace;}
    inline int get_output_size(int IMG, int K, int PAD, int STRIDE, int DILATION) {return (int)(IMG-K+2*PAD)/STRIDE + 1;}
};

#endif