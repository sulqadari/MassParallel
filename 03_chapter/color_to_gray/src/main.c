#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "picture/bmp_image.h"

#define GET_MS(start, stop)					\
	(stop.tv_sec - start.tv_sec) * 1000.0 +	\
	(stop.tv_usec - start.tv_usec) / 1000.0

static void
usage(void)
{
	fprintf(stderr, "ERROR\nUsage: ./%s "
	"<path/to/input_image.bmp> "
	"<path/to/output_image.bmp>\n", EXECUTABLE_NAME);
	exit(1);
}

int
main(int argc, char*argv[])
{
	struct timeval start, stop;
	double elapsed;
	BMP_image picture;

	if (argc < 3)
		usage();
	
	if (bmp_load_file(&picture, argv[1]))
		return (1);

	bmp_init_image(&picture);
	bmp_print_info(&picture);

	gettimeofday(&start, NULL);
	bmp_color_to_gray(&picture);
	gettimeofday(&stop, NULL);
	
	elapsed = GET_MS(start, stop);
	printf("elapsed time: %.02f ms.\n", elapsed);

	bmp_save_file(&picture, argv[2]);

	bmp_free(&picture);

	return (0);
}