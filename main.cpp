// diploma_core.cpp: определяет точку входа для консольного приложения.
//
//test
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>     
#include <helper_cuda_gl.h>   
#include <helper_functions.h>  // CUDA SDK Helper functions

#define MIN_RUNTIME_VERSION 1000
#define MIN_COMPUTE_VERSION 0x10

const static char *sSDKsample = "CUDA Test";

typedef unsigned int uint;
typedef unsigned char uchar;

const char *image_filename = "peka.bmp";
float2 transVec = { 0.0, 0.0 };

unsigned int width, height;
unsigned int  *pImg = NULL;
unsigned int *dResult = NULL;
float g = 1.0;

GLuint pbo;     // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
GLuint texid;   // texture
GLuint shader;

#define GL_TEXTURE_TYPE GL_TEXTURE_2D

extern "C" void cdTexInit(int width, int height, void *pImage);
extern "C" void cdTexFree();
extern "C" void LoadBMPFile(uchar4 **dst, unsigned int *width,
	unsigned int *height, const char *name);
extern "C" void rotate(uint* out, int w, int h, int deg);
extern "C" void translate(uint *dDest, int width, int height, float2 transVec);
extern "C" void gamma(uint *dDest, int width, int height, float g);
extern "C" void invert(uint *dDest, int width, int height);

int A = 0;
int stepA = 10;

bool isAngleCorrect(uint angle)
{
	return ((0 <= angle) && (angle <= 360)) ? true : false;
}

void display()
{


	//unsigned int *dResult;

	//checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	//size_t num_bytes;
	//checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));

	//rotTex(dResult, width, height, A);

	//checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));



	glClear(GL_COLOR_BUFFER_BIT);


	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, texid);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glDisable(GL_DEPTH_TEST);

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0, 0);
		glVertex2f(0, 0);
		glTexCoord2f(1, 0);
		glVertex2f(1, 0);
		glTexCoord2f(1, 1);
		glVertex2f(1, 1);
		glTexCoord2f(0, 1);
		glVertex2f(0, 1);
	}
	glEnd();
	glBindTexture(GL_TEXTURE_TYPE, 0);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);

	glutSwapBuffers();
	glutReportErrors();

}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case 27:
	{
		glutDestroyWindow(glutGetWindow());
		return;
		break;
	}
	case '`':
	{
		//char *command = (char *)malloc(10 * sizeof(char));
		//scanf_s("%s", command);
		//printf_s("Command: %s", command);
		break;
	}
	case 'a':
	{
		if (isAngleCorrect(A + stepA)) 
		{ 
			A += stepA; 
			checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
			size_t num_bytes;
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
			rotate(dResult, width, height, A);
			//translate(dResult, width, height, transVec);
			checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		}
		else 
		{ 
			printf("Max angle reached\n");
			A = 0;
		}
		printf_s("Deg: %d\n", A);
		break;
	}
	case 'd':
	{
		if (isAngleCorrect(A - 10))
		{
			A -= stepA;
			checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
			size_t num_bytes;
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
			rotate(dResult, width, height, A);
			//translate(dResult, width, height, transVec);
			checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		}
		else
		{
			printf("Min angle reached\n");
			A = 360;
		}
		printf_s("Deg: %d\n", A);
		break;
	}
	case 'i':
	{
		transVec.y += 1.0;
		printf("Translation vector: (%f, %f)\n", transVec.x, transVec.y);
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
		translate(dResult, width, height, transVec);
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		break;
	}
	case 'k':
	{
		transVec.y -= 1.0;
		printf("Translation vector: (%f, %f)\n", transVec.x, transVec.y);
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
		translate(dResult, width, height, transVec);
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		break;
	}
	case 'l':
	{
		transVec.x += 1.0;
		printf("Translation vector: (%f, %f)\n", transVec.x, transVec.y);
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
		translate(dResult, width, height, transVec);
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		break;
	}
	case 'j':
	{
		transVec.x -= 1.0;
		printf("Translation vector: (%f, %f)\n", transVec.x, transVec.y);
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
		translate(dResult, width, height, transVec);
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		break;
	}
	case 'g':
	{
		g = (g == 1.0) ? 2.0 : 1.0;
		printf("Gamma: %f\n", g);
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
		gamma(dResult, width, height, g);
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		break;
	}
	case 'v':
	{
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
		invert(dResult, width, height);
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		break;
	}

	default:
		break;
	}
	glutPostRedisplay();
}

void reshape(int x, int y)
{
	glViewport(0, 0, x, y);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void initCuda()
{
	cdTexInit(width, height, pImg);
}

void cleanup()
{
	if (pImg)
	{
		free(pImg);
	}

	cdTexFree();
	cudaGraphicsUnregisterResource(cuda_pbo_resource);

	glDeleteBuffersARB(1, &pbo);
	glDeleteTextures(1, &texid);
	glDeleteProgramsARB(1, &shader);
	cudaDeviceReset();
}

static const char *shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1)
	{
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		printf("Program error at position: %d\n%s\n", (int)error_pos, error_string);
		return 0;
	}

	return program_id;
}



void initGLResources()
{
	// create pixel buffer object
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte) * 4, pImg, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
		cudaGraphicsMapFlagsWriteDiscard));
	glGenTextures(1, &texid);
	glBindTexture(GL_TEXTURE_2D, texid);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}


void initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow(sSDKsample);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glewInit();
}


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

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	fprintf(stderr, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	cudaRuntimeGetVersion(&runtimeVersion);
	fprintf(stderr, "  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	fprintf(stderr, "  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);

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
		cudaDeviceProp deviceProp;
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
		fprintf(stderr, "\nNo configuration with available capabilities was found.  Test has been waived.\n");
		fprintf(stderr, "The CUDA Sample minimum requirements:\n");
		fprintf(stderr, "\tCUDA Compute Capability >= %d.%d is required\n", MIN_COMPUTE_VERSION / 16, MIN_COMPUTE_VERSION % 16);
		fprintf(stderr, "\tCUDA Runtime Version    >= %d.%d is required\n", MIN_RUNTIME_VERSION / 1000, (MIN_RUNTIME_VERSION % 100) / 10);
		exit(EXIT_WAIVED);
	}

	return bestDev;
}


int main(int argc, char **argv)
{
	int devID;
	char *ref_file = NULL;
	printf("%s Starting...\n\n", argv[0]);

	loadImageData(argc, argv);

	initGL(argc, (char **)argv);
	int dev = findCapableDevice(argc, argv);
	dev = gpuGLDeviceInit(argc, (const char **)argv);

	initCuda();
	initGLResources();
	

	glutCloseFunc(cleanup);
	printf("command mode on '`'\n");
	printf("Rotate on angle theta: rot <theta>\nTranslate on [x, y]: tran <x, y>\n");

	glutMainLoop();

	scanf_s("\n");
}