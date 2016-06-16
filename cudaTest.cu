#define _USE_MATH_DEFINES
#include<math.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>

typedef unsigned char uchar;
typedef unsigned int uint;

uint *origImg = NULL;  
uint *tempArray = NULL;
size_t pitch;

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

	float u = x - (float)width / 2;
	float v = y - (float)height / 2;

	float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
	float tv = u * sinf(theta) + v * cosf(theta) + 0.5f;

	tu += (float)width / 2;
	tv += (float)height / 2;

	outputData[y*width + x] = ((tu < width) && (tv < height) && (tu > 0) && (tv > 0)) ? rgbaFloatToInt(tex2D(rgbaTex, tu, tv)) : (uint)0;
}

__global__ void transKernel(uint *outputData, int width, int height, float2 transVec)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	float u = x + transVec.x;
	float v = y + transVec.y;
	outputData[y * width + x] = ((u < width) && (v < height) && (u > 0) && (v > 0)) ? rgbaFloatToInt(tex2D(rgbaTex, u + 0.5f, v + 0.5f)) : (uint)0;
}

__global__ void gammaKernel(uint *output, int width, int height, float g)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	float4 pixel = tex2D(rgbaTex, x + 0.5f, y + 0.5f);
	float4 gCorrected = { pow(pixel.x, g), pow(pixel.y, g), pow(pixel.z, g), pow(pixel.w, g) };

	output[y * width + x] = rgbaFloatToInt(gCorrected);
}

__global__ void invertKernel(uint *output, int width, int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	float k = 1;
	float4 pixel = tex2D(rgbaTex, x + 0.5f, y + 0.5f);
	float4 gCorrected = { k - pixel.x, k - pixel.y, k - pixel.z, k - pixel.w };

	output[y * width + x] = rgbaFloatToInt(gCorrected);
}

cudaChannelFormatDesc beforeKernelExec(uint *dDest, int width, int height)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
	checkCudaErrors(cudaBindTexture2D(0, rgbaTex, origImg, desc, width, height, pitch));
	checkCudaErrors(cudaDeviceSynchronize());
	return desc;
}


void afterKernelExec(uint *dDest, cudaChannelFormatDesc desc, int width, int height)
{
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy2D(tempArray, pitch, dDest, sizeof(int)*width, sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaBindTexture2D(0, rgbaTex, tempArray, desc, width, height, pitch));
}

extern "C"
void rotate(uint *dDest, int width, int height, int deg)
{
	float theta = (float)M_PI * (float)deg / 180.0;
	//printf("Deg angle: %d\ncos: %f\nsin: %f", deg, cosf(theta), sinf(theta));

	cudaChannelFormatDesc desc = beforeKernelExec(dDest, width, height);

	dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	rotKernel <<< gridSize, blockSize >>>(dDest, width, height, theta);

	afterKernelExec(dDest, desc, width, height);
}

extern "C"
void translate(uint *dDest, int width, int height, float2 transVec)
{
	cudaChannelFormatDesc desc = beforeKernelExec(dDest, width, height);

	dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	transKernel << < gridSize, blockSize >> >(dDest, width, height, transVec);

	afterKernelExec(dDest, desc, width, height);
}

extern "C"
void gamma(uint *dDest, int width, int height, float g)
{
	cudaChannelFormatDesc desc = beforeKernelExec(dDest, width, height);

	dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	gammaKernel<<< gridSize, blockSize >>>(dDest, width, height, g);

	afterKernelExec(dDest, desc, width, height);
}

extern "C"
void invert(uint *dDest, int width, int height)
{
	cudaChannelFormatDesc desc = beforeKernelExec(dDest, width, height);

	dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	invertKernel << < gridSize, blockSize >> >(dDest, width, height);

	afterKernelExec(dDest, desc, width, height);
}

extern "C"
void cdTexInit(int width, int height, uint *hImage)
{
	// copy image data to array
	checkCudaErrors(cudaMallocPitch(&origImg, &pitch, sizeof(uint)*width, height));
	checkCudaErrors(cudaMallocPitch(&tempArray, &pitch, sizeof(uint)*width, height));
	checkCudaErrors(cudaMemcpy2D(origImg, pitch, hImage, sizeof(uint)*width,
		sizeof(uint)*width, height, cudaMemcpyHostToDevice));
}

extern "C"
void cdTexFree()
{
	checkCudaErrors(cudaFree(origImg));
	checkCudaErrors(cudaFree(tempArray));
}