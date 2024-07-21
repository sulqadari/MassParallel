#ifndef VECTOR_CUDA_H
#define VECTOR_CUDA_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define REG_MODIFIER register

typedef struct Vector_t {
	int32_t length;
	int32_t current;
	uint8_t* buff;
} Vector;

uint8_t push_value(Vector* vec, uint8_t value);
uint8_t pop_value(Vector* vec);
uint8_t get_value(Vector* vec);
uint8_t init_vector(Vector* vec, uint32_t length);
void free_vector(Vector* vec);
int8_t cmp_length(Vector* first, Vector* second);
void add_vectors(Vector* first, Vector* second, Vector* result);

#endif /* VECTOR_CUDA_H */