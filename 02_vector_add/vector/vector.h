#ifndef VECTOR_CUDA_H
#define VECTOR_CUDA_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#if defined (USE_CUDA)
	#define CUDA_GLOBAL	__global__
	#define CUDA_HOST	__host__
	#define CUDA_DEVICE	__device__
	#define CUDA_HD		__host__ __device__

	#define CUDA_ASSERT_ERROR(status_)					\
	do {												\
		if (status_ != cudaSuccess) {					\
			printf("ERROR: %s\nsource: %s\nline: %d\n",	\
					cudaGetErrorString(status_),		\
					__FILE__, __LINE__);				\
			break;									\
		}												\
	} while (0)

#else
	#define CUDA_ASSERT_ERROR(status_)
	#define CUDA_GLOBAL
	#define CUDA_HOST
	#define CUDA_DEVICE
	#define CUDA_HD
#endif

typedef struct Vector_t {
	int32_t length;
	int32_t current;
	uint8_t* buff;
} Vector;

CUDA_HD uint8_t push_value(Vector* vec, uint8_t value);
CUDA_HD uint8_t pop_value(Vector* vec);
CUDA_HD uint8_t get_value(Vector* vec);
CUDA_HOST uint8_t init_vector(Vector* vec, uint32_t length);
CUDA_HOST void free_vector(Vector* vec);
CUDA_HD int8_t cmp_length(Vector* first, Vector* second);
CUDA_DEVICE void add_vectors(uint8_t* first, uint8_t* second, uint8_t* result, int32_t length);
CUDA_GLOBAL void cuda_main(uint8_t* first, uint8_t* second, uint8_t* result, int32_t length, uint32_t count);
#endif /* VECTOR_CUDA_H */