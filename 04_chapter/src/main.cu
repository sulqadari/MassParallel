#include <stdio.h>
#include <stdint.h>

#define CUDA_ASSERT_ERROR(status_)					\
do {												\
	if (status_ != cudaSuccess) {					\
		printf("ERROR: %s\nfile: %s\nline: %d\n",	\
				cudaGetErrorString(status_),		\
				__FILE__, __LINE__);				\
		exit(1);									\
	}												\
} while (0)

__host__ void
displayProperties(cudaDeviceProp* props, int32_t idx)
{
	printf(
		"\n"
		"IDX:                                    %02d\n"
		"VERSION:                                %d.%d\n"
		"NAME:                                   %s\n"
		"TOTAL GLOBAL MEMORY:                    %04lu MB\n"
		"TOTAL CONST MEMORY:                     %04lu bytes\n"
		"SHARED MEM AVAILABLE TO A THREAD BLOCK: %04lu bytes\n"
		"REGISTERS  AVAILABLE TO A THREAD BLOCK: %04d\n"
		"WARP SIZE:                              %04d threads\n"
		"MAX THREADS PER BLOCK:                  %04d\n"
		"MAX SIZE OF BLOCK DIMS:\n"
		"                                      X = %d;\n"
		"                                      Y = %d;\n"
		"                                      Z = %d;\n"
		"MAX SIZE OF GRID DIMS:\n"
		"                                      X = %d;\n"
		"                                      Y = %d;\n"
		"                                      Z = %d;\n"
		"CLOCK RATE:                             %04d (KHz)\n"
		"\n",
		idx,
		props->major,props->minor,
		props->name,

		props->totalGlobalMem / (1024 * 1000),
		props->totalConstMem,
		props->sharedMemPerBlock,

		props->regsPerBlock,
		props->warpSize,
		props->maxThreadsPerBlock,
		props->maxThreadsDim[0],props->maxThreadsDim[1],props->maxThreadsDim[2],
		props->maxGridSize[0],  props->maxGridSize[1],  props->maxGridSize[2],
		props->clockRate
	);
}

int
main(int argc, char* argv[])
{
	int32_t gpu_count = 0;
	cudaDeviceProp properties;

	cudaGetDeviceCount(&gpu_count);
	cudaDeviceSynchronize();
	CUDA_ASSERT_ERROR(cudaGetLastError());

	/* Iterate through all avaiable GPUs and print theirs properties. */
	for (int32_t i = 0; i < gpu_count; ++i) {

		cudaGetDeviceProperties(&properties, i);
		displayProperties(&properties, i);
	}

}
