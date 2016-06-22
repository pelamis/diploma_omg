#ifndef OPENGL_CUDA_H_
#define OPENGL_CUDA_H_

#include "commons.h"
#include "callbacks.h"

extern GLuint pbo;
extern struct cudaGraphicsResource *cuda_pbo_resource;

extern GLuint vbo;
extern struct cudaGraphicsResource *cuda_vbo_resource;

void initGL(int argc, char **argv);
void initGLResources();
void vboCreate(GLuint* vbo, cudaGraphicsResource **vboRes, uint vboResFlags);
void vboDelete(GLuint *vbo, cudaGraphicsResource *vboRes);
void initCuda();
void cleanup();

#endif

