
#include "conv.cuh"


Conv2d::Conv2d(
                /*BATCH_SIZE*/int N,
                /*INPUT_CHANNEL*/int IN_C,
                /*OUTPUT_CHANNEL*/int OUT_C,
                /*IMAGE_HEIGHT*/int IMG_H,
                /*IMAGE_WIDTH*/int IMG_W,
                /*KERNEL_HEIGHT*/int K_H,
                /*KERNEL_WIDTH*/int K_W,
                /*PAD_H*/int PAD_H,
                /*PAD_W*/int PAD_W,
                /*STRIDE_H*/int STRIDE_H,
                /*STRIDE_W*/int STRIDE_W,
                /*DILATION_H*/int DILATION_H,
                /*DILATION_W*/int DILATION_W
) {

    /** CAUTION **/ 
    /** Only support a single precision operation and NCHW data layout **/

    // Context
    cudnnErrChk (cudnnCreate (&handle));
    
    // Input
    cudnnErrChk (cudnnCreateTensorDescriptor (&desc_input));
    cudnnErrChk (cudnnSetTensor4dDescriptor (desc_input, 
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/CUDNN_DATA_FLOAT, /*N*/N, /*C*/ IN_C, /*H*/IMG_H, /*W*/IMG_W));

    // Output
    cudnnErrChk (cudnnCreateTensorDescriptor (&desc_output));
    cudnnErrChk (cudnnSetTensor4dDescriptor (desc_output, 
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/CUDNN_DATA_FLOAT, /*N*/N, /*C*/ OUT_C, /*H*/IMG_H, /*W*/IMG_W));

    // Kernel weights
    cudnnErrChk (cudnnCreateFilterDescriptor (&desc_kernel));
    cudnnErrChk (cudnnSetFilter4dDescriptor (desc_kernel, 
        /*DATATYPE*/CUDNN_DATA_FLOAT, /*LAYOUT*/CUDNN_TENSOR_NCHW, /*O_C*/OUT_C, /*I_C*/IN_C, /*K_H*/K_H, /*K_W*/K_W));

    // Layer description
    cudnnErrChk (cudnnCreateConvolutionDescriptor (&desc_layer));
    cudnnErrChk (cudnnSetConvolution2dDescriptor (desc_layer, 
        /*PAD_H*/PAD_H, /*PAD_W*/PAD_W, /*STRIDE_VERTICAL*/STRIDE_H, /*STRIDE_HORIZONTAL*/STRIDE_W, /*DILATION_H*/DILATION_H, /*DILATION_W*/DILATION_W, /*MODE*/CUDNN_CROSS_CORRELATION, /*DATATYPE*/CUDNN_DATA_FLOAT));
    cudnnErrChk (cudnnGetConvolutionForwardAlgorithm (
        handle, desc_input, desc_kernel, desc_layer, desc_output, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_algo));

    // Workspace allocation
    cudnnErrChk (cudnnGetConvolutionForwardWorkspaceSize (handle, desc_input, desc_kernel, desc_layer, desc_output, conv_algo, &size_workspace));
    printf("Workspace allocated (size: %.3f MB)\n", ((float)size_workspace)*1e-6);
    cudaErrChk (cudaMalloc (&d_workspace, size_workspace));

    // Allocate kernel weights and output
    cudaErrChk (cudaMalloc (&d_kernel, sizeof(float)*OUT_C*IN_C*K_H*K_W));
    BATCH_SIZE = N;
    O_C = OUT_C;
    O_H = get_output_size(IMG_H, K_H, PAD_H, STRIDE_H, DILATION_H);
    O_W = get_output_size(IMG_W, K_W, PAD_W, STRIDE_W, DILATION_W);
    
}



Conv2d::~Conv2d() {

    cudaErrChk (cudaFree (d_kernel));
    cudaErrChk (cudaFree (d_workspace));

    cudnnErrChk (cudnnDestroyTensorDescriptor (desc_input));
    cudnnErrChk (cudnnDestroyTensorDescriptor (desc_output));
    cudnnErrChk (cudnnDestroyFilterDescriptor (desc_kernel));
    cudnnErrChk (cudnnDestroyConvolutionDescriptor (desc_layer));
    cudnnErrChk (cudnnDestroy(handle));
    
}


float* Conv2d::run(const float* d_input) {

    float * d_output = nullptr; // output must be freed (!!)
    cudaErrChk (cudaMalloc (&d_output, sizeof(float)*BATCH_SIZE*O_C*O_H*O_W));
    const float alpha=1, beta=0;
    cudnnErrChk (cudnnConvolutionForward (handle
                                        , /*ALPHA*/&alpha
                                        , /*INPUT*/desc_input, d_input
                                        , /*KERNEL*/desc_kernel, d_kernel
                                        , /*LAYER*/desc_layer, conv_algo, d_workspace, size_workspace
                                        , /*BETA*/&beta
                                        , /*OUTPUT*/desc_output, d_output
                                    ))



    return d_output;
}

