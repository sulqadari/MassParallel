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

static const char*
get_version(int32_t major, int32_t minor)
{
	switch (major) {
		case 2: return "Fermi";
		case 3: return "Kepler";
		
		case 5: return "Maxwell";
		case 6: return "Pascal";
		case 7: return "Volta/Turing";
		case 8: return "Ampere";
		case 9: return "Hopper";
		default: return "Unknow Device type";
	}
}

static int32_t
count_cores(int32_t MPs, int32_t major, int32_t minor)
{
	switch (major) {
		case 2: return minor == 1 ? MPs * 48 : MPs * 32;	// Fermi
		case 3: return MPs * 192;	// Kepler
		case 5: return MPs * 128;	// Maxwell
		case 6: {	// Pascal
			if (minor == 1 || minor == 2)
				return MPs * 128;
			else if (minor == 0)
				return MPs * 64;
			else
				return (0);
		}
		case 7: {	// Volta and Turing
			if (minor == 0 || minor == 5)
				return MPs * 64;
			else
				return (0);
		}
		case 8: {	// Ampere
			if (minor == 0)
				return MPs * 64;
			else if (minor == 6 || minor == 9)
				return MPs * 128;
			else
				return (0);
		}
		case 9: return minor == 0 ? MPs * 128 : 0;
		default:  return (0);
	}
}

__host__ void
displayProperties(cudaDeviceProp* props, int32_t idx)
{
	printf(
		"\n"
		"                1. Common\n"
		"version:                                       %d.%d %s\n"
		"Device name:                                   %s\n"
		"CUDA cores:                                    %d\n"
		
		"\n                2. Memory\n"
		"32-bit registers available per block:          %04d \n"
		"Shared memory available per block:             %.01f Kb\n"
		"32-bit registers available per multiprocessor: %04d \n"
		"Shared memory available per multiprocessor:    %.01f Kb\n"
		"Size of L2 cache                               %.01f Mb\n"
		"Constant memory available on device:           %.01f Kb\n"
		"Global memory available on device:             %.01f Gb\n"
		"Global memory bus width                        %d bits\n"

		"\n                3. Compute Capability\n"
		"Warp size:                                    %d threads\n"
		"Maximum number of threads per block:          %d\n"
		"Maximum resident threads per multiprocessor:  %d\n"
		"Number of multiprocessors on device:          %d\n"

		"Maximum size of each dimension of a block:\n"
		"                                         X =  %d\n"
		"                                         Y =  %d\n"
		"                                         Z =  %d\n"
		"Maximum size of each dimension of a grid:\n"
		"                                         X =  %d\n"
		"                                         Y =  %d\n"
		"                                         Z =  %d\n"
		"\n",
		props->major,props->minor,
		get_version(props->major,props->minor),
		props->name,
		count_cores(props->multiProcessorCount, props->major,props->minor),

		props->regsPerBlock,
		(float)(props->sharedMemPerBlock / 1024),
		props->regsPerMultiprocessor,
		(float)(props->sharedMemPerMultiprocessor / 1024),
		(float)(props->l2CacheSize / (1024 * 1000)),
		(float)(props->totalConstMem / 1024),
		(float)(props->totalGlobalMem / (1024 * 1000000)),
		props->memoryBusWidth,

		props->warpSize,
		props->maxThreadsPerBlock,
		props->maxThreadsPerMultiProcessor,
		props->multiProcessorCount,

		props->maxThreadsDim[0],props->maxThreadsDim[1],props->maxThreadsDim[2],
		props->maxGridSize[0],  props->maxGridSize[1],  props->maxGridSize[2]
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
