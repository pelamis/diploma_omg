#ifndef COMMONS_H_
#define COMMONS_H_

#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>     
#include <helper_cuda_gl.h>   
#include <helper_functions.h>

#define MIN_RUNTIME_VERSION 1000
#define MIN_COMPUTE_VERSION 0x10

#define GL_TEXTURE_TYPE GL_TEXTURE_2D

typedef unsigned int uint;
typedef unsigned char uchar;

typedef struct Image {
	unsigned int *data;
	unsigned int *width;
	unsigned int *height;
}Image;

extern float2 transVec;

extern unsigned int width, height;
extern unsigned int  *pImg;
extern unsigned int *dResult;
extern float g;

extern cudaDeviceProp deviceProp;

////OpenGL pixel buffer object
//extern GLuint pbo;
//extern struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//
////OpenGL vertex buffer object
//extern GLuint vbo;
//extern struct cudaGraphicsResource *cuda_vbo_resource;


extern GLuint texid;   // texture
extern GLuint shader;

#endif