#include "vector/vector.h"

uint8_t
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

uint8_t
pop_value(Vector* vec)
{
	if (vec->current <= 0) {
		/* Vector is empty.
		 * Just in case, set the 'current' to zero. */
		return vec->buff[vec->current = 0];
	}
	
	return vec->buff[--vec->current];
}

uint8_t
get_value(Vector* vec)
{
	/* The corner case: the buffer is empty, thus return a value at index zero. */
	if (vec->current == 0) {
		return vec->buff[vec->current];
	} else {
		return vec->buff[vec->current - 1];
	}
	
}

int8_t
is_empty(Vector* vec)
{
	return vec->current <= 0 ? 1 : 0;
}

int8_t
is_full(Vector* vec)
{
	return vec->current >= vec->length ? 1 : 0;
}

uint8_t
init_vector(Vector* vec, uint32_t length)
{
	vec->buff = (uint8_t*)malloc(length);
	if (NULL == vec->buff)
		return (1);
	
	vec->length = length;
	vec->current = 0;

	return (0);
}

void
free_vector(Vector* vec)
{
	free(vec->buff);
}

int8_t
cmp_length(Vector* first, Vector* second)
{
	if (first->length > second->length)
		return (1);
	
	if (first->length < second->length)
		return (-1);
	
	return (0);
}

void
add_vectors(REG_MODIFIER Vector* first, REG_MODIFIER Vector* second, REG_MODIFIER Vector* result)
{
	uint32_t length = 0;

	/* Use the lesser length value to avoid array boundary violation. */
	if (cmp_length(first, second) < 0) {
		length = first->length;
	} else {
		length = second->length;
	}

	for (REG_MODIFIER uint32_t i = 0; i < length; ++i) {
		result->buff[i] = first->buff[i] + second->buff[i];
	}
}