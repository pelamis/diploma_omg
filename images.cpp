#include "images.h"
#include <stdio.h>
#pragma comment(lib, "User32.lib")

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#pragma pack(1)

std::list<Image> series;
const char *inputDir = "input";
using namespace std;

int loadSeries(list<Image> *inputSeries)
{
	HANDLE file = INVALID_HANDLE_VALUE;
	Image curImg;
	LARGE_INTEGER filesize;
	WIN32_FIND_DATA fileFindData;
	TCHAR szDir[MAX_PATH];
	StringCchCopy(szDir, MAX_PATH, inputDir);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	file = FindFirstFile(szDir, &fileFindData);

	while (FindNextFile(file, &fileFindData) != 0)
	{
		if (fileFindData.dwFileAttributes != 16) //check if not a directory
		{
			loadBMP(&curImg, fileFindData.cFileName);
			inputSeries->push_back(curImg);
			printf_s("File %s added\n", fileFindData.cFileName);
		}
		
	}
	return 0;
}

void imgCleanup(list<Image> *inputSeries)
{
	Image curImg;
	int i;
	if (!(inputSeries->empty()))
	{
		for (curImg = inputSeries->back(); inputSeries->size() > 1; curImg = inputSeries->back())
		{
			free(curImg.data);
			inputSeries->pop_back();
		}

		curImg = inputSeries->back();
		free(curImg.data);
		inputSeries->clear();
	}

}

int loadBMP(Image *dest, const char *name)
{
	BMPHeader header;
	BMPInfoHeader infoHeader;
	unsigned int row = 0; 
	unsigned int col = 0;
	FILE *f = NULL;

	printf_s("Loading %s...\n", name);

	//f = fopen(name, "rb");

	if (!(f = fopen(name, "rb")))
	{
		printf_s("ERROR: file access denied\n");
		return -1;
	}

	fread(&header, sizeof(header), 1, f);

	if (header.bfType != 0x4d42) {
		printf_s("ERROR: bad file signature\n");
		return -1;
	}

	fread(&infoHeader, sizeof(infoHeader), 1, f);

	if (infoHeader.bitCount != 24)
	{
		printf_s("ERROR: %hd-bit color depth is not supported (only 24-bit)\n");
		return -1;
	}

	if (infoHeader.compression != 0)
	{
		printf_s("ERROR: compressed images are not supported\n");
		return -1;
	}
	
	dest -> height = infoHeader.height;
	dest -> width = infoHeader.width;
	dest->data = (uchar4*)malloc(4 * infoHeader.width * infoHeader.height);

	fseek(f, header.bfOffBits - sizeof(header) - sizeof(infoHeader), SEEK_CUR);

	for (row = 0; row < dest->height; row++)
	{
		for (col = 0; col < dest->width; col++)
		{
			dest->data[row * dest->width + col].w = 0;
			dest->data[row * dest->width + col].z = fgetc(f);
			dest->data[row * dest->width + col].y = fgetc(f);
			dest->data[row * dest->width + col].x = fgetc(f);

		}

		//alignment skip
		for (col = 0; col < (unsigned int)(4 - (3 * infoHeader.width) % 4) % 4; col++) fgetc(f);
	}

	if (ferror(f))
	{
		printf_s("ERROR: unknown error\n");
		free(dest->data);
		return -1;
	}
	else 
	{
		printf_s("File loaded\n");
	}
	fclose(f);
	return 0;
}