#include "opengl_init.h"

static const char *shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

const static char *sSDKsample = "CUDA Test";

GLuint compileASMShader(GLenum program_type, const char *code)
{
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1)
	{
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		printf("Program error at position: %d\n%s\n", (int)error_pos, error_string);
		return 0;
	}

	return program_id;
}

void initGLResources()
{
	// create pixel buffer object
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte) * 4, pImg, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
		cudaGraphicsMapFlagsWriteDiscard));
	glGenTextures(1, &texid);
	glBindTexture(GL_TEXTURE_2D, texid);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow(sSDKsample);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutTimerFunc(REFRESH, timerEvent, 0);
	glewInit();
}
