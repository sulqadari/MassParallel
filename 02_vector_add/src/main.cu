#include "vector/vector.h"
#include <string.h>
#include <sys/time.h>

#define GET_MS(start, stop)					\
	(stop.tv_sec - start.tv_sec) * 1000.0 +	\
	(stop.tv_usec - start.tv_usec) / 1000.0

#define VECTOR_SIZE (10000 * 1024)
#define VECTORS_ARRAY_SIZE (3)
#define CUDA_BLOCK_DIM 256

static void
set_random(Vector* vec)
{
	uint32_t val = 0;
	while (!push_value(vec, val++)) { }
}


static int8_t
init_vectors(Vector* vectors)
{
	for (uint32_t i = 0; i < VECTORS_ARRAY_SIZE; ++i) {
		if (init_vector(&vectors[i], VECTOR_SIZE))
			return (1);
	}

	return (0);
}

static void
free_vectors(Vector* vectors)
{
	for (uint32_t i = 0; i < VECTORS_ARRAY_SIZE; ++i)
		free_vector(&vectors[i]);
}

int
main(int argc, char*argv[])
{
	struct timeval start, stop;
	double elapsed;
	uint32_t length = 0;
	cudaError_t err = cudaSuccess;
	Vector vectors[VECTORS_ARRAY_SIZE];
	
	uint8_t *dev_first, *dev_second, *dev_result;

	if (init_vectors(vectors)) {
		printf("ERROR: failed to initialized one of the vectors.\n");
		goto _free_vectors;
	}

	set_random(&vectors[0]);
	set_random(&vectors[1]);

	cudaMalloc(&dev_first, vectors[0].length * sizeof(uint8_t));
	cudaDeviceSynchronize(); err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("ERROR: %s\nsource: %s\nline: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(1);
	}

	cudaMalloc(&dev_second, vectors[1].length * sizeof(uint8_t));
	cudaDeviceSynchronize(); err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("ERROR: %s\nsource: %s\nline: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(1);
	}

	cudaMalloc(&dev_result, vectors[2].length * sizeof(uint8_t));
	cudaDeviceSynchronize(); err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("ERROR: %s\nsource: %s\nline: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(1);
	}

	if (cmp_length(&vectors[0], &vectors[1]) < 0) {
		length = vectors[0].length;
	} else {
		length = vectors[1].length;
	}

	for (uint32_t i = 0; i < 10; ++i) {
		
		gettimeofday(&start, NULL);
		
		for (volatile uint32_t j = 0; j < 1000; ++j) {
			add_vectors<<<ceil(length / (double)CUDA_BLOCK_DIM), CUDA_BLOCK_DIM>>>(dev_first, dev_second, dev_result, length);
			cudaDeviceSynchronize(); err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("ERROR: %s\nsource: %s\nline: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
			}
		}
		
		gettimeofday(&stop, NULL);

		elapsed = GET_MS(start, stop);
		printf("%02d) elapsed time: %.04f ms.\n", i + 1, elapsed);
	}

_free_vectors:
	free_vectors(vectors);
	cudaFree(dev_first);
	cudaFree(dev_second);
	cudaFree(dev_result);

	return (0);
}