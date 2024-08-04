#ifndef _CUDA_COMMON_H
#define _CUDA_COMMON_H

#define GET_MS(start, stop)					\
	(stop.tv_sec - start.tv_sec) * 1000.0 +	\
	(stop.tv_usec - start.tv_usec) / 1000.0


#define CUDA_ASSERT_ERROR()							\
do {												\
	cudaDeviceSynchronize();						\
	cudaError_t status_ = cudaGetLastError();		\
	if (status_ != cudaSuccess) {					\
		printf("ERROR: %s\nfile: %s\nline: %d\n",	\
				cudaGetErrorString(status_),		\
				__FILE__, __LINE__);				\
		break;										\
	}												\
} while (0)

#endif /* _CUDA_COMMON_H */