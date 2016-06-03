#ifndef OPENGL_INIT_H_
#define OPENGL_INIT_H_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>

void displayCallback();
void kbdCallback(unsigned char key, int x, int y);
void initializeGL(int argc, char** argv);
void buffersInit();