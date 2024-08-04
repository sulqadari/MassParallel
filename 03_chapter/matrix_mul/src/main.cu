#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include "common.h"

__host__ static void
matrix_print(int32_t* mtx, uint32_t len, const char* label)
{
#if defined (PRINT_DEBUG)
	printf("\n%s\n", label);
	for (uint32_t row = 0; row < len; ++row) {
		printf("\t\t");
		for (uint32_t col = 0; col < len; ++col) {
			printf("%02d ", mtx[row * len + col]);
		}
		printf("\n");
	}
#endif /* PRINT_DEBUG */
}

__host__ static void
matrix_set_random(int32_t* mtx, uint32_t len)
{
	for (uint32_t row = 0; row < len; ++row)
		for (uint32_t col = 0; col < len; ++col) {
			mtx[row * len + col] = rand() % 10;
		}
}

__host__ static void
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

__global__ void
kern_matrix_multiply(int32_t* first, int32_t* second, int32_t* output, uint32_t width)
{
	uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t value = 0;

	if (row >= width || col >= width)
		return;
	
	for (uint32_t k = 0; k < width; ++k)
		value += first[row * width + k] * second[k * width + col];
	
	output[row * width + col] = value;
}

int
main(int argc, char* argv[])
{
	uint32_t dim_ = 2;
	uint32_t mem_size = ((dim_ * dim_) * sizeof(int32_t));

	int32_t* first = NULL;
	int32_t* second = NULL;
	int32_t* output = NULL;

	int32_t* dev_first = NULL;
	int32_t* dev_second = NULL;
	int32_t* dev_output = NULL;

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

	cudaMalloc(&dev_first,  mem_size);
	CUDA_ASSERT_ERROR();
	cudaMalloc(&dev_second, mem_size);
	CUDA_ASSERT_ERROR();
	cudaMalloc(&dev_output, mem_size);
	CUDA_ASSERT_ERROR();

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

	cudaMemcpy(dev_first, first, mem_size, cudaMemcpyHostToDevice);
	CUDA_ASSERT_ERROR();
	cudaMemcpy(dev_second, second, mem_size, cudaMemcpyHostToDevice);
	CUDA_ASSERT_ERROR();

	dim3 grid_(ceil(mem_size / (double)32), 1, 1);
	dim3 block_(32, 32, 1);

	/* Start point. */
	gettimeofday(&start, NULL);

	kern_matrix_multiply<<<grid_, block_>>>(dev_first, dev_second, dev_output, dim_);
	CUDA_ASSERT_ERROR();

	/* logging result. */
	gettimeofday(&stop, NULL);

	elapsed = GET_MS(start, stop);
	printf("elapsed time: %.02f ms.\n\n", elapsed);
	
	cudaMemcpy(output, dev_output, mem_size, cudaMemcpyDeviceToHost);
	CUDA_ASSERT_ERROR();

	matrix_print(output, dim_, "matrix C:");

	/* deallocating memory. */
	free(first);
	free(second);
	free(output);

	cudaFree(dev_first);	CUDA_ASSERT_ERROR();
	cudaFree(dev_second);	CUDA_ASSERT_ERROR();
	cudaFree(dev_output);	CUDA_ASSERT_ERROR();

	return (0);
}