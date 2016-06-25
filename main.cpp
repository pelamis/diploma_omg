// diploma_core.cpp: определяет точку входа для консольного приложения.
//
//test
#include <Windows.h>
#include "callbacks.h"
#include "opengl_cuda.h"

const char *image_filename = "test.bmp";
unsigned int width, height;
unsigned int  *pImg = NULL;
unsigned int *dResult = NULL;

extern "C" void cdTexInit(int width, int height, void *pImage);
extern "C" void cdTexFree();

// Rewrite this function!
extern "C" void LoadBMPFile(uchar4 **dst, unsigned int *width,
	unsigned int *height, const char *name);

void loadImageData(int argc, char **argv)
{
	// load image (needed so we can get the width and height before we create the window
	char *image_path = NULL;

	image_path = sdkFindFilePath(image_filename, argv[0]);

	if (image_path == NULL)
	{
		fprintf(stderr, "Error finding image file '%s'\n", image_filename);
		exit(EXIT_FAILURE);
	}
	
	LoadBMPFile((uchar4 **)&pImg, &width, &height, image_path);

	if (pImg == NULL)
	{
		fprintf(stderr, "Error opening file '%s'\n", image_path);
		exit(EXIT_FAILURE);
	}

}

bool checkCUDAProfile(int dev, int min_runtime, int min_compute)
{
	int runtimeVersion = 0;

	cudaGetDeviceProperties(&deviceProp, dev);

	fprintf(stderr, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	cudaRuntimeGetVersion(&runtimeVersion);
	fprintf(stderr, "  CUDA Runtime Version              :\t%d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	fprintf(stderr, "  CUDA Compute Capability           :\t%d.%d\n", deviceProp.major, deviceProp.minor);
	fprintf(stderr, "  CUDA max 2D texture dimensions    :\t%d x %d\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
	fprintf(stderr, "  CUDA maxThreadsDim                :\t%d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	fprintf(stderr, "  CUDA maxThreadsPerBlock           :\t%d\n", deviceProp.maxThreadsPerBlock);
	fprintf(stderr, "  CUDA maxThreadsPerMultiProcessor  :\t%d\n", deviceProp.maxThreadsPerMultiProcessor);
	fprintf(stderr, "  CUDA maxGridSize                  :\t%d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
		deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

	if (runtimeVersion >= min_runtime && ((deviceProp.major << 4) + deviceProp.minor) >= min_compute)
	{
		return true;
	}
	else
	{
		return false;
	}
}

int findCapableDevice(int argc, char **argv)
{
	int dev;
	int bestDev = -1;

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0)
	{
		fprintf(stderr, "There are no CUDA capable devices.\n");
	}
	else
	{
		fprintf(stderr, "Found %d CUDA Capable device(s) supporting CUDA\n", deviceCount);
	}

	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaGetDeviceProperties(&deviceProp, dev);

		if (checkCUDAProfile(dev, MIN_RUNTIME_VERSION, MIN_COMPUTE_VERSION))
		{
			fprintf(stderr, "\nFound CUDA Capable Device %d: \"%s\"\n", dev, deviceProp.name);

			if (bestDev == -1)
			{
				bestDev = dev;
				fprintf(stderr, "Setting active device to %d\n", bestDev);
			}
		}
	}

	if (bestDev == -1)
	{
		fprintf(stderr, "\nNo configuration with available capabilities was found.\n");
		fprintf(stderr, "The CUDA Sample minimum requirements:\n");
		fprintf(stderr, "\tCUDA Compute Capability >= %d.%d is required\n", MIN_COMPUTE_VERSION / 16, MIN_COMPUTE_VERSION % 16);
		fprintf(stderr, "\tCUDA Runtime Version    >= %d.%d is required\n", MIN_RUNTIME_VERSION / 1000, (MIN_RUNTIME_VERSION % 100) / 10);
		exit(EXIT_WAIVED);
	}

	return bestDev;
}


int main(int argc, char **argv)
{
	char *ref_file = NULL;
	printf("%s Starting...\n\n", argv[0]);

	loadImageData(argc, argv);

	initGL(argc, (char **)argv);
	int dev = findCapableDevice(argc, argv);
	dev = gpuGLDeviceInit(argc, (const char **)argv);

	initCuda();
	initGLResources();

	printf("Rotate: r\nTranslate: t\nGamma: g\nInvert: i\n");

	glutMainLoop();

	scanf_s("\n");
}