#include "images.h"
#include <stdio.h>
#pragma comment(lib, "User32.lib")

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#pragma pack(1)

//std::vector<Image> series;
const char *inputDir = "input";
const char *outputDir = "output";
using namespace std;

int loadSeries(vector<Image> *inputSeries)
{
	HANDLE file = INVALID_HANDLE_VALUE;
	Image curImg;
	//LARGE_INTEGER filesize;
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




void imgCleanup(vector<Image> *inputSeries)
{
	//Image curImg;
	int i;

	if (!(inputSeries->empty()))
	{
		for (i = 0; i < inputSeries->size(); i++)
		{
			free((*inputSeries)[i].name);
			free((*inputSeries)[i].header);
			free((*inputSeries)[i].infoHeader);
			free((*inputSeries)[i].data);
		}
		inputSeries->clear();
	}

}

int writeBMP(Image *src)
{
	HANDLE hOut;
	FILE *of;
	size_t padding;
	int width = src->infoHeader->width;
	int height = src->infoHeader->height;
	int i, j;
	unsigned int pix;
	TCHAR path[MAX_PATH];
	StringCchCopy(path, MAX_PATH, outputDir);
	StringCchCat(path, MAX_PATH, TEXT("\\OUT"));
	StringCchCat(path, MAX_PATH, src->name);
	of = fopen(path, "w+b");
	fwrite(src->header, sizeof(BMPHeader), 1, of);
	fwrite(src->infoHeader, sizeof(BMPInfoHeader), 1, of);

	padding = ((width * 3) % 4) ? 4 - (width * 3) % 4 : 0;
	RGB pix3;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			uchar4 pix4 = src->data[i * width + j];
			pix3.z = pix4.z;
			pix3.y = pix4.y;
			pix3.x = pix4.x;
			fwrite(&pix3, sizeof(RGB), 1, of);
		}
		pix3.x = 0, pix3.y = 0; pix3.z = 0;
		if (padding != 0) {
			fwrite(&pix3, padding, 1, of);
		}
	}
	fclose(of);
	//hOut = CreateFile(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	//if (hOut == INVALID_HANDLE_VALUE) {
	//	printf("ERROR %x \n", GetLastError());
	//	return -1;
	//}

	//CloseHandle(hOut);
	return 0;
}


int loadBMP(Image *dest, const char *name)
{
	BMPHeader *hdr = (BMPHeader *)malloc(sizeof(BMPHeader)); //dest->header;
	BMPInfoHeader *infoHdr = (BMPInfoHeader *)malloc(sizeof(BMPInfoHeader)); //dest->infoHeader;
	int width, height;
	int row = 0; 
	int col = 0;
	FILE *f = NULL;

	printf_s("Loading %s...\n", name);

	if (!(f = fopen(name, "rb")))
	{
		printf_s("ERROR: file access denied\n");
		return -1;
	}

	fread(hdr, sizeof(BMPHeader), 1, f);

	if (hdr->bfType != 0x4d42) {
		printf_s("ERROR: bad file signature\n");
		return -1;
	}

	fread(infoHdr, sizeof(BMPInfoHeader), 1, f);

	if (infoHdr->bitCount != 24)
	{
		printf_s("ERROR: %hd-bit color depth is not supported (only 24-bit)\n");
		return -1;
	}

	if (infoHdr->compression != 0)
	{
		printf_s("ERROR: compressed images are not supported\n");
		return -1;
	}
	size_t nsize = strnlen_s(name, MAX_PATH - 10);
	dest->name = (char *)malloc(nsize + 1);
	strncpy_s(dest->name, nsize + 1, name, nsize);
	dest->name[nsize] = 0;
	width = infoHdr->width;
	height = infoHdr->height;
	dest->header = hdr;
	dest->infoHeader = infoHdr;
	dest->data = (uchar4*)malloc(4 * infoHdr->width * infoHdr->height);

	fseek(f, hdr->bfOffBits - sizeof(BMPHeader) - sizeof(BMPInfoHeader), SEEK_CUR);

	for (row = 0; row < height; row++)
	{
		for (col = 0; col < width; col++)
		{
			dest->data[row * width + col].w = 0;
			dest->data[row * width + col].z = fgetc(f);
			dest->data[row * width + col].y = fgetc(f);
			dest->data[row * width + col].x = fgetc(f);

		}

		//alignment skip
		for (col = 0; col < (unsigned int)(4 - (3 * infoHdr->width) % 4) % 4; col++) fgetc(f);
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