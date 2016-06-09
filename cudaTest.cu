#define _USE_MATH_DEFINES
#include <math.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


typedef unsigned char uchar;
typedef unsigned int uint;

uint *dImage  = NULL;   //original image
uint *dTemp   = NULL; 

// uint *originalImg = NULL;
// uint *tempArray = NULL;

size_t pitch;
texture<float, 2, cudaReadModeElementType> tex;
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;

__global__ void VecAdd(float* A, float* B, float* C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] + B[i];
}

__global__ void rotKernel(uint *outputData,
                                int width,
                                int height,
                                int deg)
{
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	int cf = 0, sf = 0;
	if ((deg == 0) || (deg == 360)) cf = 1, sf = 0;
	if (deg == 90) cf = 0, sf = 1;
	if (deg == 180) cf = -1, sf = 0;
	if (deg == 270) cf = 0, sf = -1;

    int u = x - (int)width/2; 
    int v = y - (int)height/2; 
    int tu = u*cf - v*sf; 
    int tv = v*cf + u*sf; 

    tu /= width; 
    tv /= height; 

    outputData[y*width + x] = tex2D(tex, tu + (int)1/2, tv + (int)1/2);
}

extern "C"
void rotTex(uint *dDest, int width, int height, int deg)
{

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dImage, desc, width, height, pitch));

    checkCudaErrors(cudaDeviceSynchronize());

    dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
    dim3 blockSize(16, 16);
    rotKernel<<< gridSize, blockSize>>>(dDest, width, height, deg);

    checkCudaErrors(cudaDeviceSynchronize());
                                                                                      
    checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width, sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, width, height, pitch));
    return;
}

extern "C"
void cudaTest(uint* out, int w, int h, int deg)
{
	//float theta = (float)M_PI * (float)deg / 180.0;
	printf("Deg angle: %d\n",deg);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dImage, desc, w, h, pitch));

	dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(w / dimBlock.x, h / dimBlock.y, 1);
	rotKernel<<<dimGrid,dimBlock,0>>>(out, w, h, deg);

	cudaDeviceSynchronize();
	
	checkCudaErrors(cudaMemcpy2D(dTemp, pitch, out, sizeof(int)*w, sizeof(int)*w, h, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, w, h, pitch));
	//VecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}

/*extern "C"
void cudaTextureInit(int height, int width, uint *img)
{
	cudaMallocPitch(&originalImg, &pitch, sizeof(uint)*width, height);
	cudaMallocPitch(&tempArray, &pitch, sizeof(uint)*width, height);
	cudaMemcpy2D(originalImg, pitch, img, sizeof(uint)*width, sizeof(uint)*width, height, cudaMemcpyHostToDevice);
}

extern "C"
void cudaTexturesFree()
{
	cudaFree(originalImg);
	cudaFree(tempArray);
}*/


extern "C"
void cdTexInit(int width, int height, uint *hImage)
{
    // copy image data to array
    checkCudaErrors(cudaMallocPitch(&dImage, &pitch, sizeof(uint)*width, height));
    checkCudaErrors(cudaMallocPitch(&dTemp,  &pitch, sizeof(uint)*width, height));
    checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint)*width,
                                 sizeof(uint)*width, height, cudaMemcpyHostToDevice));
}

extern "C"
void cdTexFree()
{
    checkCudaErrors(cudaFree(dImage));
    checkCudaErrors(cudaFree(dTemp));
}