#ifndef CALLBACKS_H_
#define CALLBACKS_H_

#include "commons.h"
#include "cudakernels.h"

#define REFRESH 10

void display();
void timerEvent(int value);
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void reshape(int x, int y);
#endif