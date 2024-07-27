#include "vector/vector.h"

CUDA_HD uint8_t
push_value(Vector* vec, uint8_t value)
{
	if (vec->current >= vec->length) {
		/* Vector is full.
		 * Just in case, assign 'current' the 'length' value. */
		vec->current = vec->length;
		return (1);
	}
	
	vec->buff[vec->current++] = value;
	return (0);
}

CUDA_HD uint8_t
pop_value(Vector* vec)
{
	if (vec->current <= 0) {
		/* Vector is empty.
		 * Just in case, set the 'current' to zero. */
		return vec->buff[vec->current = 0];
	}
	
	return vec->buff[--vec->current];
}

CUDA_HD uint8_t
get_value(Vector* vec)
{
	/* The corner case: the buffer is empty, thus return a value at index zero. */
	if (vec->current == 0) {
		return vec->buff[vec->current];
	} else {
		return vec->buff[vec->current - 1];
	}
	
}

CUDA_HD int8_t
is_empty(Vector* vec)
{
	return vec->current <= 0 ? 1 : 0;
}

CUDA_HD int8_t
is_full(Vector* vec)
{
	return vec->current >= vec->length ? 1 : 0;
}

CUDA_HOST uint8_t
init_vector(Vector* vec, uint32_t length)
{
	vec->buff = (uint8_t*)malloc(length);
	if (NULL == vec->buff)
		return (1);
	
	vec->length = length;
	vec->current = 0;

	return (0);
}

CUDA_HOST void
free_vector(Vector* vec)
{
	free(vec->buff);
}

CUDA_HD int8_t
cmp_length(Vector* first, Vector* second)
{
	if (first->length > second->length)
		return (1);
	
	if (first->length < second->length)
		return (-1);
	
	return (0);
}

CUDA_DEVICE void
add_vectors(uint8_t* first, uint8_t* second, uint8_t* result, int32_t length)
{
	volatile int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < length)
		result[i] += first[i] + second[i];
}

CUDA_GLOBAL void
cuda_main(uint8_t* first, uint8_t* second, uint8_t* result, int32_t length, uint32_t count)
{
	for (volatile uint32_t j = 0; j < count; ++j)
		add_vectors(first, second, result, length);
}