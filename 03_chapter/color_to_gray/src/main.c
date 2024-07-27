#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "picture/BMPicture.h"

#define GET_MS(start, stop)					\
	(stop.tv_sec - start.tv_sec) * 1000.0 +	\
	(stop.tv_usec - start.tv_usec) / 1000.0

static void
usage(void)
{
	fprintf(stderr, "Usage: %s <path/to/image.bmp>\n", EXECUTABLE_NAME);
	exit(1);
}

static void
upload_image(const char* path, BMPicture* image)
{
	FILE* file;
	size_t bytesRead;

	file = fopen(path, "rb");
	if (NULL == file) {
		fprintf(stderr, "Couldn't open source file '%s'.\n", path);
		exit(74);
	}

	fseek(file, 0L, SEEK_END);	/* Move file prt to EOF. */
	image->size = ftell(file);		/* How far we are from start of the file? */
	rewind(file);				/* Rewind file ptr back to the beginning. */

	image->buff = malloc(image->size);
	if (NULL == image->buff) {
		fprintf(stderr, "Failed to allocate memory for image->buff"
		"for source file '%s'.\n", path);
		exit(74);
	}

	bytesRead = fread(image->buff, sizeof(char), image->size, file);
	if (bytesRead < image->size) {
		fprintf(stderr, "Couldn't read source file '%s'.\n", path);
		exit(74);
	}

	fclose(file);
}

static void
color_to_gray_convertion(void)
{

}

int
main(int argc, char*argv[])
{
	if (argc < 2)
		usage();
	
	BMPicture BMPicture;
	upload_image(argv[1], &BMPicture);


	return (0);
}