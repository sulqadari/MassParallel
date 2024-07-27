#include "vector/vector.h"
#include <string.h>
#include <sys/time.h>

#define GET_MS(start, stop)					\
	(stop.tv_sec - start.tv_sec) * 1000.0 +	\
	(stop.tv_usec - start.tv_usec) / 1000.0

#define VECTOR_SIZE			(10000 * 1024)
#define VECTORS_ARRAY_SIZE	(3)
#define CUDA_BLOCK_DIM		256

static void
set_random(Vector* vec)
{
	uint32_t val = 1;
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
	uint32_t length = 0, count = 1000;
	Vector vectors[VECTORS_ARRAY_SIZE];
	uint8_t *dev_first, *dev_second, *dev_result;

	if (argc > 1)
		count = strtoull(argv[1], NULL, 10);

	if (init_vectors(vectors)) {
		printf("ERROR: failed to initialized one of the vectors.\n");
		goto _free_vectors;
	}

	set_random(&vectors[0]);
	set_random(&vectors[1]);

	cudaMalloc(&dev_first, vectors[0].length * sizeof(uint8_t));
	cudaDeviceSynchronize();
	CUDA_ASSERT_ERROR(cudaGetLastError(), _free_vectors);


	cudaMalloc(&dev_second, vectors[1].length * sizeof(uint8_t));
	cudaDeviceSynchronize();
	CUDA_ASSERT_ERROR(cudaGetLastError(), _free_vectors);

	cudaMalloc(&dev_result, vectors[2].length * sizeof(uint8_t));
	cudaDeviceSynchronize();
	CUDA_ASSERT_ERROR(cudaGetLastError(), _free_vectors);

	if (cmp_length(&vectors[0], &vectors[1]) < 0) {
		length = vectors[0].length;
	} else {
		length = vectors[1].length;
	}
	
	cudaMemcpy(dev_first, vectors[0].buff, length, cudaMemcpyHostToDevice);
	CUDA_ASSERT_ERROR(cudaGetLastError(), _free_vectors);
	
	cudaMemcpy(dev_second, vectors[1].buff, length, cudaMemcpyHostToDevice);
	CUDA_ASSERT_ERROR(cudaGetLastError(), _free_vectors);

	printf("\nBefore: ");
	for (uint32_t i = 0; i < 16; ++i)
		printf("%02X ", vectors[2].buff[i]);
	
	printf("\n");

	gettimeofday(&start, NULL);
	

	cuda_main<<<ceil(length / (double)CUDA_BLOCK_DIM), CUDA_BLOCK_DIM>>>(dev_first,
																		dev_second,
																		dev_result,
																		length,
																		count);
	cudaDeviceSynchronize();
	gettimeofday(&stop, NULL);

	CUDA_ASSERT_ERROR(cudaGetLastError(), _free_vectors);

	elapsed = GET_MS(start, stop);
	printf("\nelapsed time: %.02f ms.\n", elapsed);

	cudaMemcpy(vectors[2].buff, dev_result, length, cudaMemcpyDeviceToHost);
	CUDA_ASSERT_ERROR(cudaGetLastError(), _free_vectors);

	printf("\nAfter:  ");
	for (uint32_t i = 0; i < 16; ++i)
		printf("%02X ", vectors[2].buff[i]);
	
	printf("\n");

_free_vectors:
	cudaFree(dev_first);
	cudaFree(dev_second);
	cudaFree(dev_result);
	free_vectors(vectors);

	return (0);
}