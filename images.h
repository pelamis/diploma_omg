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

typedef struct
{
	unsigned char x, y, z;
} RGB;



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

typedef struct Image {
	char *name;
	BMPHeader *header;
	BMPInfoHeader *infoHeader;
	//unsigned int width, height;
	uchar4 *data;
}Image;

extern std::vector<Image> series;

int loadBMP(Image *dest, const char *name);
int writeBMP(Image *src);
int loadSeries(std::vector<Image> *inputSeries);
void imgCleanup(std::vector<Image> *inputSeries);
#endif