#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>

#define GET_MS(start, stop)	\
	(stop.tv_sec - start.tv_sec) * 1000.0 + \
	(stop.tv_usec - start.tv_usec) / 1000.0

static void
initArray(float* array, int length, float value)
{
	for (int i = 0; i < length; ++i)
		array[i] = value;
}

static __global__ void
add(float* addendum, float* addend, int length)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index < length)
		addend[index] = addendum[index] + addend[index];
}

int
main(int argc, char* argv[])
{
	struct timeval start, stop;
	double elapsed;
	int length = 1 << 24;	// 16 777 216 ; 29 == 536 870 912

	if (argc > 1)
		length = 1 << strtol(argv[1], NULL, 10);		
	
	float* x = (float*)malloc(length * sizeof(float));
	float* y = (float*)malloc(length * sizeof(float));
	
	initArray(x, length, 1.0f);
	initArray(y, length, 2.0f);

	float* d_x;
	float* d_y;

	if (cudaMalloc(&d_x, length * sizeof(float))) {
		fprintf(stderr, "Failed to allocate memory for d_x\n");
		free(x);
		free(y);
		return (1);
	}

	if (cudaMalloc(&d_y, length * sizeof(float))) {
		fprintf(stderr, "Failed to allocate memory for d_y\n");
		free(x);
		free(y);
		cudaFree(d_x);
		return (1);
	}

	if (cudaMemcpy(d_x, x, length, cudaMemcpyHostToDevice)) {
		fprintf(stderr, "Failed to copy from x to d_x\n");
		free(x);
		free(y);
		cudaFree(d_x);
		cudaFree(d_y);
		return (1);
	}

	if (cudaMemcpy(d_y, y, length, cudaMemcpyHostToDevice)) {
		fprintf(stderr, "Failed to copy from y to d_y\n");
		free(x);
		free(y);
		cudaFree(d_x);
		cudaFree(d_y);
		return (1);
	}

	int blockSize = 256;
	unsigned int roundedSize = ceil(length / (double)blockSize);

	gettimeofday(&start, NULL);
	add<<<roundedSize, blockSize>>>(d_x, d_y, length);
	cudaDeviceSynchronize();

	gettimeofday(&stop, NULL);
	elapsed = GET_MS(start, stop);
	printf("elapsed time: %.04f ms.\n", elapsed);

	free(x);
	free(y);
	cudaFree(d_x);
	cudaFree(d_y);
	return (0);
}