#include <iostream>
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



int main(int argc, char** argv) {

    cudnnHandle_t cudnn;
    cudnnErrChk (cudnnCreate (&cudnn));

    /***
     *** 1. Describing operands : Input, output tensors and filter (kernel, weights)
     ***/
    
    // Input tensor
    cudnnTensorDescriptor_t input_desc;
    cudnnErrChk (cudnnCreateTensorDescriptor (&input_desc));
    cudnnErrChk (cudnnSetTensor4dDescriptor (input_desc, 
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/CUDNN_DATA_FLOAT, /*N*/1, /*C*/ 3, /*H*/128, /*W*/128));

    // Output tensor
    cudnnTensorDescriptor_t output_desc;
    cudnnErrChk (cudnnCreateTensorDescriptor (&output_desc));
    cudnnErrChk (cudnnSetTensor4dDescriptor (output_desc, 
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/CUDNN_DATA_FLOAT, /*N*/1, /*C*/ 3, /*H*/128, /*W*/128));

    // Kernel == Filter == Weights
    cudnnFilterDescriptor_t kernel_desc;
    cudnnErrChk (cudnnCreateFilterDescriptor (&kernel_desc));
    cudnnErrChk (cudnnSetFilter4dDescriptor (kernel_desc, 
        /*DATATYPE*/CUDNN_DATA_FLOAT, /*LAYOUT*/CUDNN_TENSOR_NCHW, /*O_C*/3, /*I_C*/3, /*K_H*/3, /*K_W*/3));

    
    /***
     *** 2. Describing the convolution kernel
     ***/
    
    // Convolution layer
    cudnnConvolutionDescriptor_t convolution_desc;
    cudnnErrChk (cudnnCreateConvolutionDescriptor (&convolution_desc));
    cudnnErrChk (cudnnSetConvolution2dDescriptor (convolution_desc, 
        /*PAD_H*/1, /*PAD_W*/1, /*STRIDE_VERTICAL*/1, /*STRIDE_HORIZONTAL*/1, /*DILATION_H*/1, /*DILATION_W*/1, /*MODE*/CUDNN_CROSS_CORRELATION, /*DATATYPE*/CUDNN_DATA_FLOAT));

    // Convolution algorithm
    cudnnConvolutionFwdAlgo_t convolution_algo;
    cudnnErrChk (cudnnGetConvolutionForwardAlgorithm (cudnn, input_desc, kernel_desc, convolution_desc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algo));

    /***
     *** 3. Get work space size
     ***/
    size_t bytes_workspace = 0;
    cudnnErrChk (cudnnGetConvolutionForwardWorkspaceSize (cudnn, input_desc, kernel_desc, convolution_desc, output_desc, convolution_algo, &bytes_workspace));
    printf("Workspace size: %.3f MB\n", ((float)bytes_workspace)*1e-6);
 

    /***
     *** 4. Allocating memories
     ***/
    void *d_workspace = nullptr;
    cudaErrChk (cudaMalloc (&d_workspace, bytes_workspace));
    float *h_input=nullptr, *h_output=nullptr, *h_kernel=nullptr;
    float *d_input=nullptr, *d_output=nullptr, *d_kernel=nullptr;

    h_input = (float*) malloc (1*3*128*128*sizeof(float));
    h_output = (float*) malloc (1*3*128*128*sizeof(float));
    h_kernel = (float*) malloc (3*3*3*3*sizeof(float));
    cudaErrChk (cudaMalloc (&d_input, 1*3*128*128*sizeof(float)));
    cudaErrChk (cudaMalloc (&d_output, 1*3*128*128*sizeof(float)));
    cudaErrChk (cudaMalloc (&d_kernel, 3*3*3*3*sizeof(float)));

    /***
     *** 5. Launch the kernel
     ***/
    const float alpha=1, beta=0;
    cudnnErrChk (cudnnConvolutionForward (cudnn
                                        , /*ALPHA*/&alpha
                                        , /*INPUT*/input_desc, d_input
                                        , /*KERNEL*/kernel_desc, d_kernel
                                        , /*LAYER*/convolution_desc, convolution_algo, d_workspace, bytes_workspace
                                        , /*BETA*/&beta
                                        , /*OUTPUT*/output_desc, d_output
                                    ))


    /***
     *** 6. Get the result
     ***/
    cudaErrChk (cudaMemcpy (h_output, d_output, 1*3*128*128*sizeof(float), cudaMemcpyDeviceToHost));



    /*** 
     *** 7. Finalize : Free all objects
     ***/
    free (h_input);
    free (h_output);
    free (h_kernel);

    cudaErrChk (cudaFree (d_workspace));
    cudaErrChk (cudaFree (d_input));
    cudaErrChk (cudaFree (d_output));
    cudaErrChk (cudaFree (d_kernel));

    cudnnErrChk (cudnnDestroyTensorDescriptor (input_desc));
    cudnnErrChk (cudnnDestroyTensorDescriptor (output_desc));
    cudnnErrChk (cudnnDestroyFilterDescriptor (kernel_desc));
    cudnnErrChk (cudnnDestroyConvolutionDescriptor (convolution_desc));
    cudnnErrChk (cudnnDestroy(cudnn));
    return 0;
}
