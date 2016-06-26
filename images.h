#ifndef IMAGES_H_
#define IMAGES_H_

#include "commons.h"
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#pragma pack(1)

//typedef struct
//{
//	unsigned char x, y, z, w;
//} uchar4;

typedef struct Image {
	unsigned int width, height;
	uchar4 *data;
}Image;

typedef struct
{
	short bfType;
	int bfSize;
	short bfReserved1;
	short bfReserved2;
	int bfOffBits;
} BMPHeader;

typedef struct
{
	int size;
	int width;
	int height;
	short planes;
	short bitCount;
	unsigned compression;
	unsigned sizeImage;
	int xPelsPerMeter;
	int yPelsPerMeter;
	int clrUsed;
	int clrImportant;
} BMPInfoHeader;



extern std::list<Image> series;

int loadBMP(Image *dest, const char *name);
int loadSeries(std::list<Image> *inputSeries);
#endif