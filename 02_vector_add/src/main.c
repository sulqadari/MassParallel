#include "vector/vector.h"
#include <string.h>
#include <sys/time.h>

#define GET_MS(start, stop)					\
	(stop.tv_sec - start.tv_sec) * 1000.0 +	\
	(stop.tv_usec - start.tv_usec) / 1000.0

#define VECTOR_SIZE (10000 * 1024)
#define VECTORS_ARRAY_SIZE (3)

static void
set_random(Vector* vec)
{
	uint32_t val = 0;
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
	uint32_t length = 0;
	Vector vectors[VECTORS_ARRAY_SIZE];

	if (init_vectors(vectors)) {
		printf("ERROR: failed to initialized one of the vectors.\n");
		goto _free_vectors;
	}

	set_random(&vectors[0]);
	set_random(&vectors[1]);

	/* Use the lesser length value to avoid array boundary violation. */
	if (cmp_length(&vectors[0], &vectors[1]) < 0) {
		length = vectors[0].length;
	} else {
		length = vectors[1].length;
	}

	for (uint32_t i = 0; i < 10; ++i) {
		
		gettimeofday(&start, NULL);
		
		for (uint32_t j = 0; j < 1000; ++j)
			add_vectors(&vectors[0], &vectors[1], &vectors[2], length);
		
		gettimeofday(&stop, NULL);

		elapsed = GET_MS(start, stop);
		printf("%02d) elapsed time: %.04f ms.\n", i + 1, elapsed);
	}

_free_vectors:
	free_vectors(vectors);

	return (0);
}