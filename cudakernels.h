#ifndef CUDAKERNELS_H_
#define CUDAKERNELS_H_

#include "commons.h"

extern "C" void cdTexInit(int width, int height, void *pImage);
extern "C" void cdTexFree();

extern "C" void rotate(uint* out, int w, int h, int deg);
extern "C" void translate(uint *dDest, int width, int height, float2 transVec);
extern "C" void gamma(uint *dDest, int width, int height, float g);
extern "C" void invert(uint *dDest, int width, int height);
extern "C" void rotate2(uint* src, uint *out, int w, int h, int deg);

#endif