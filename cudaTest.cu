#define _USE_MATH_DEFINES
#include<math.h>
#include <helper_math.h>
//#include <vector_types.h>
#include <helper_functions.h>
//#include <cuda_runtime.h>
#include <helper_cuda.h>


typedef unsigned char uchar;
typedef unsigned int uint;

uint *dImage = NULL;   //original image
uint *dTemp = NULL;

// uint *originalImg = NULL;
// uint *tempArray = NULL;

size_t pitch;
texture<float, 2, cudaReadModeElementType> tex;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> rgbaTex;

__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(fabs(rgba.y));
	rgba.z = __saturatef(fabs(rgba.z));
	rgba.w = __saturatef(fabs(rgba.w));
	return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__global__ void rotKernel(uint *outputData,
	int width,
	int height,
	float theta)
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	//cos t -sin t
	//sin t cos t
	float u = x * cosf(theta) - y * sinf(theta);
	float v = x * sinf(theta) + y * cosf(theta);
	if (u < 0) u = width - u;
	if (v < 0) v = height - v;
	//float u = (float)width - x; //- (float)width / 2;
	//float v = (float)height - y; // -(float)height / 2;
	//float tu = - v;
	//float tv = u;

	//tu /= (float)width;
	//tv /= (float)height;

	outputData[y*width + x] = rgbaFloatToInt(tex2D(rgbaTex, u, v));
}

extern "C"
void rotTex(uint *dDest, int width, int height, int deg)
{
	float theta = (float)M_PI * (float)deg / 180.0;
	printf("Deg angle: %d\ncos: %f\nsin: %f", deg, cosf(theta), sinf(theta));
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
	checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dImage, desc, width, height, pitch));

	checkCudaErrors(cudaDeviceSynchronize());

	dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	rotKernel << < gridSize, blockSize >> >(dDest, width, height, theta);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width, sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, width, height, pitch));
	return;
}

extern "C"
void cdTexInit(int width, int height, uint *hImage)
{
	// copy image data to array
	checkCudaErrors(cudaMallocPitch(&dImage, &pitch, sizeof(uint)*width, height));
	checkCudaErrors(cudaMallocPitch(&dTemp, &pitch, sizeof(uint)*width, height));
	checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint)*width,
		sizeof(uint)*width, height, cudaMemcpyHostToDevice));
}

extern "C"
void cdTexFree()
{
	checkCudaErrors(cudaFree(dImage));
	checkCudaErrors(cudaFree(dTemp));
}