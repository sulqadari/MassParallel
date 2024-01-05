#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>

#define GET_MS(start, stop) \
	(stop.tv_sec - start.tv_sec) * 1000.0 + \
	(stop.tv_usec - start.tv_usec) / 1000.0

static void
toHex(unsigned char* array, uint32_t length, const char* path)
{
	FILE* file;

	file = fopen(path, "wb");
	if (NULL == file) {
		fprintf(stderr, "Failed to create hex dump '%s'.\n", path);
		return;
	}

	for (uint32_t i = 0; i < length; ++i) {
		if (i != 0 && ((i % 16) == 0))
			fprintf(file, "\n");

		fprintf(file, "%02X ", array[i]);
	}

	fclose(file);
}

static void
writeFile(const char* path, unsigned char* image, size_t imageSize)
{
	FILE* file;
	size_t bytesWritten;

	file = fopen(path, "wb");
	if (NULL == file) {
		fprintf(stderr, "Failed to create file '%s'.\n", path);
		return;
	}

	bytesWritten = fwrite(image, sizeof(unsigned char), imageSize, file);
	if (bytesWritten < imageSize)
		fprintf(stderr, "Failed to write data into '%s'.\n", path);

	fclose(file);
}

static unsigned char*
readFile(const char* path, size_t* fileSize)
{
	FILE* file;
	unsigned char* buffer;
	size_t bytesRead;

	file = fopen(path, "rb");
	if (NULL == file) {
		fprintf(stderr, "Failed to open image file '%s'.\n", path);
		return NULL;
	}

	fseek(file, 0L, SEEK_END);
	*fileSize = ftell(file);
	rewind(file);

	buffer = malloc(*fileSize + 1);
	if (NULL == buffer) {
		fprintf(stderr, "Failed to allocate memory for buffer.\n");
		return NULL;
	}

	bytesRead = fread(buffer, sizeof(char), *fileSize, file);
	if (*fileSize > bytesRead) {
		fprintf(stderr, "Failed to copy the file into buffer.\n");
		free(buffer);
		fclose(file);
		return NULL;
	}

	buffer[*fileSize] = '\0';
	fclose(file);

	return buffer;
}

static void
toGrayscale(unsigned char* image, uint32_t imageSize,
								unsigned char* output)
{
	uint32_t offBits = 10;
	offBits = image[offBits + 1] << 8 | image[offBits];

	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char gray;

	for (uint32_t offset = offBits; offset < imageSize; offset += 3) {
		r = image[offset];
		g = image[offset + 1];
		b = image[offset + 2];
		
		gray = (0.21f * r) + (0.71f * g) + (0.07f * b);
		output[offset] = gray;
		output[offset + 1] = gray;
		output[offset + 2] = gray;
	}
}

int
main(int argc, char* argv[])
{
	// struct timeval start, stop;
	// double elapsed;
	if (argc < 2) {
		printf("Usage: FadeOutCPU <~/path/to/image/picture.bmp>\n");
		return (1);
	}

	unsigned char* image;
	size_t imageSize = 0;

	// download an image (.bmp format)
	image = readFile(argv[1], &imageSize);
	if (NULL == image)
		return (1);
	
	unsigned char* output = malloc(imageSize);

	// copy the header and BitInfo
	memcpy(output, image, 64);

	// perform color to grayscale convertion
	toGrayscale(image, imageSize, output);

	// compile the new block of image with the old
	// metadata and store the output on the disk.
	toHex(output, imageSize, "./hexDump.txt");
	writeFile("grayscaled.bmp", output, imageSize);
	free(image);
	free(output);

	return (0);
}