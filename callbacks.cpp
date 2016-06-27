#include "callbacks.h"
#include "opengl_cuda.h"
#include "images.h"

bool animate = false;
bool invertF = false;
int A = 0;
int stepA = 10;
float g = 1.0;
float2 transVec = { 0.0, 0.0 };

void switchImage(std::vector<Image> *imgArray, int index)
{

}

void skipGarbageInput()
{
	fseek(stdin, 0, SEEK_END);
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH, timerEvent, 0);
	}
}

void reshape(int x, int y)
{
	int pcside;
	pcside = (width>height ? width : height) * 2;
	glViewport(0, 0, x, y);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);
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
		break;
	}
	case 'r':
	{
		size_t num_bytes;
		skipGarbageInput();
		printf_s("Angle (deg): ");
		scanf_s("%d", &A);
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
		//rotate2(dSrc, dResult, width, height, A);
		rotate(dResult, width, height, A);
		//translate(dResult, width, height, transVec);
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		break;
	}
	case 'c':
	{
		printf_s("Restoring to default...\n");
		A = 0;
		transVec.x = 0.0f;
		transVec.y = 0.0f;
		g = 1.0f;
		break;
	}
	case 'a':
	{
		char rotDirection = 0x00;
		skipGarbageInput();
		if (!animate)
		{
			printf_s("Rotate direction: (l)eft / (r)ight: ");
			scanf_s("%c", &rotDirection);
			printf_s("\n");
			switch (rotDirection)
			{
			case 'r':
			{
				stepA = -(int)abs(stepA);
				break;
			}
			case 'l':
			{
				stepA = (int)abs(stepA);
				break;
			}
			default:
			{
				printf_s("Wrong symbol: %c.\n Direction set to default (left).\n", rotDirection);
				stepA = (int)abs(stepA);
				break;
			}
			}

			animate = true;
		}
		else {
			printf_s("Animation stopped\n");
			animate = false;
		}
		break;
	}
	case 't':
	{
		skipGarbageInput();
		printf_s("Translate vector: <x> <y>) (deg): ");
		scanf_s("%f %f", &transVec.x, &transVec.y);
		//checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		//size_t num_bytes;
		//checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
		////rotate(dResult, width, height, A);
		//translate(dResult, width, height, transVec);
		//checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));


	case 'g':
	{
		g = (g == 1.0f) ? 2.0f : 1.0f;
		printf("Gamma: %f\n", g);
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
		gamma(dResult, width, height, g);
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		break;
	}
	case 'i':
	{

		//invertF = (invertF) ? false : true;
		//printf_s("invert: %d \n", invertF);
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
}

void display()
{
	unsigned int *dSrc;
	//unsigned int *dResult;
	float *vertexes;
	size_t num_bytes;

	//if (animate)
	//{
	//	A = (A + 1) % 360;
	//}

	//applyTransformations();
	//checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	//checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource));
	//rotate2(dSrc, dResult, width, height, A);
	//rotate(dResult, width, height, A);
	//translate(dResult, width, height, transVec);
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
		glVertex2f(-0.5, -0.5);
		glTexCoord2f(1, 0);
		glVertex2f(0.5, -0.5);
		glTexCoord2f(1, 1);
		glVertex2f(0.5, 0.5);
		glTexCoord2f(0, 1);
		glVertex2f(-0.5, 0.5);
	}
	glEnd();
	glBindTexture(GL_TEXTURE_TYPE, 0);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);

	glutSwapBuffers();
	glutReportErrors();

}