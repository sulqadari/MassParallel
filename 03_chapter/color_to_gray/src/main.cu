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
	BMP_Info* info;

	uint8_t* pixels_d;
	if (argc < 3)
		usage();
	
	if (bmp_load_file(&picture, argv[1]))
		return (1);

	bmp_init_image(&picture);
	bmp_print_info(&picture);

	info = &picture.info;

	do {
		cudaMalloc(&pixels_d, picture.info.image_size);
		cudaDeviceSynchronize();
		CUDA_ASSERT_ERROR(cudaGetLastError());

		cudaMemcpy(pixels_d, &picture.buff[picture.info.header.pixels_offset],
					picture.info.image_size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		CUDA_ASSERT_ERROR(cudaGetLastError());

		gettimeofday(&start, NULL);

		dim3 grid_(ceil(picture.info.pic_width / 16.0), ceil(picture.info.pic_height / 16.0), 1);
		dim3 block_(16, 16, 1);

		bmp_color_to_gray<<<grid_, block_>>>(pixels_d,
											info->pic_width,
											info->pic_height,
											info->bit_count);
		cudaDeviceSynchronize();
		CUDA_ASSERT_ERROR(cudaGetLastError());

		gettimeofday(&stop, NULL);

		elapsed = GET_MS(start, stop);
		printf("elapsed time: %.02f ms.\n", elapsed);

		cudaMemcpy(&picture.buff[picture.info.header.pixels_offset], pixels_d,
					picture.info.image_size, cudaMemcpyDeviceToHost);
		bmp_save_file(&picture, argv[2]);
	} while (0);

	cudaFree(pixels_d);

	bmp_free(&picture);

	return (0);
}