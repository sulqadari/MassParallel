#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>

#include "common.h"

static void
matrix_print(int32_t* mtx, uint32_t len, const char* label)
{
#if defined (PRINT_DEBUG)
	printf("\n%s\n", label);
	for (uint32_t row = 0; row < len; ++row) {
		printf("\t\t");
		for (uint32_t col = 0; col < len; ++col) {
			printf("%03d ", mtx[row * len + col]);
		}
		printf("\n");
	}
#endif /* PRINT_DEBUG */
}

static void
matrix_set_random(int32_t* mtx, uint32_t len)
{
	int32_t idx = 0;
	for (uint32_t row = 0; row < len; ++row)
		for (uint32_t col = 0; col < len; ++col) {
			idx = row * len + col;
			mtx[idx] = (rand() % 10);
		}
}

static void
hostMalloc(int32_t** mtx, uint32_t len)
{
	*mtx = (int32_t*) malloc(len);

	if (NULL == *mtx) {
		fprintf(stderr, "Failed to allocate memory.\n"
						"file: %s\nline: %d\n",
						__FILE__, __LINE__);
		exit(1);
	}
}

static void
host_matrix_multiply(int32_t* first, int32_t* second, int32_t* output, uint32_t width)
{
	int32_t f_idx = 0;
	int32_t s_idx = 0;
	int32_t o_idx = 0;

	for (uint32_t row = 0; row < width; ++row) {
		for (uint32_t col = 0; col < width; ++col) {

			int32_t value = 0;
			
			for (uint32_t k = 0; k < width; ++k){
				
				f_idx = row * width + k;
				s_idx = k * width + col;
				value += first[f_idx] * second[s_idx];
			}
			
			o_idx = row * width + col;
			output[o_idx] = value;
		}
	}
}

int
main(int argc, char* argv[])
{
	uint32_t dim_ = 2;
	uint32_t mem_size = ((dim_ * dim_) * sizeof(int32_t));

	int32_t* first = NULL;
	int32_t* second = NULL;
	int32_t* output = NULL;

	struct timeval start, stop;
	double elapsed;

	if (argc > 1) {
		dim_ = strtoull(argv[1], NULL, 10);
		mem_size = ((dim_ * dim_) * sizeof(int32_t));
	}

	/* Allocating memory */
	hostMalloc(&first,  mem_size);
	hostMalloc(&second, mem_size);
	hostMalloc(&output, mem_size);

	/* Initialization. */
	srand(time(NULL));
	matrix_set_random(first, dim_);
	matrix_set_random(second, dim_);
	
	printf(
		"Matrix A: %.02f MB\n"
		"Matrix B: %f MB\n",
		(float)mem_size / (1024 * 1000),
		(float)mem_size / (1024 * 1000)
	);

	matrix_print(first, dim_, "matrix A:");
	matrix_print(second, dim_, "matrix B:");

	/* Start point. */
	gettimeofday(&start, NULL);

	host_matrix_multiply(first, second, output, dim_);

	/* logging result. */
	gettimeofday(&stop, NULL);

	elapsed = GET_MS(start, stop);
	printf("elapsed time: %.02f ms.\n\n", elapsed);

	matrix_print(output, dim_, "matrix C:");

	/* deallocating memory. */
	free(first);
	free(second);
	free(output);

	return (0);
}