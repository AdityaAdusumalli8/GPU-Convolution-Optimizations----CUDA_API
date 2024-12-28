#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16
__constant__ float mask[15000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
   /*
   Modify this function to implement the forward pass described in Chapter 16.
   We have added an additional dimension to the tensors to support an entire mini-batch
   The goal here is to be correct AND fast.


   Function paramter definitions:
   output - output
   input - input
   mask - convolution kernel
   B - batch_size (number of images in x)
   M - number of output feature maps
   C - number of input feature maps
   H - input height dimension
   W - input width dimension
   K - kernel height and width (K x K)
   S - stride step length
   */


   const int H_out = (H - K)/S + 1;
   const int W_out = (W - K)/S + 1;


   // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
   // An example use of these macros:
   // float a = in_4d(0,0,0,0)
   // out_4d(0,0,0,0) = a


   #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
   #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
   #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


   int tempWidth = (ceil(W_out/(1.0*TILE_WIDTH)));


   int bx = blockIdx.x;
   int by = blockIdx.y;


   int h = (blockIdx.z/tempWidth)*TILE_WIDTH+threadIdx.y;
   int w = (blockIdx.z%tempWidth)*TILE_WIDTH+threadIdx.x;


   if (h < H_out && w < W_out && by<M)
   {
       float value = 0.0f;
       for (int c = 0; c < C; c++) {
           for (int p = 0; p < K; p++) {
               for (int q = 0; q < K; q++) {
                   value += in_4d(bx, c, h*S+p, w*S+q) * mask_4d(by, c, p, q);
               }
           }
       }
       out_4d(blockIdx.x, by, h, w) = value;
   }

   #undef out_4d
   #undef in_4d
   #undef mask_4d
}




__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    int input_size = B*C* H *W* sizeof(float);
    int mask_size = M*C* K *K* sizeof(float);
    int output_size = B*M* H_out *W_out* sizeof(float);

    cudaMalloc((void **)device_output_ptr, output_size);
    cudaMalloc((void **)device_input_ptr, input_size);
    cudaMalloc((void **)device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input,  input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask,mask_size, cudaMemcpyHostToDevice);
    
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    dim3 blockDim(16, 16,1); 
    dim3 gridDim(B,M,(ceil(W_out/ (float)TILE_WIDTH) * ceil(H_out/ (float)TILE_WIDTH)));

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;   
    int output_size = B* M* H_out*W_out *sizeof(float) ;
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}