#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdint.h>

#define GET_MS(start, stop)	\
	(stop.tv_sec - start.tv_sec) * 1000.0 + \
	(stop.tv_usec - start.tv_usec) / 1000.0

static void
initArray(float* array, int length, float value)
{
	for (int i = 0; i < length; ++i)
		array[i] = value;
}

static void
add(float* addendum, float* addend, int length)
{
	for (int i = 0; i < length; ++i) {
		addend[i] = addendum[i] + addend[i];
	}
}

static int
hasError(float* resBuff, int length)
{
	for (int i = 0; i < length; ++i) {
		if (resBuff[i] != 3.0f)
			return (i);
	}

	return (-1);
}

int
main(int argc, char* argv[])
{
	struct timeval start, stop;
	double elapsed;
	int length = 1 << 24;	// 16 777 216

	if (argc > 1)
		length = 1 << strtol(argv[1], NULL, 10);		
	
	float* x = (float*)malloc(length * sizeof(float));
	float* y = (float*)malloc(length * sizeof(float));

	initArray(x, length, 1.0f);
	initArray(y, length, 2.0f);

	gettimeofday(&start, NULL);
	add(x, y, length);
	gettimeofday(&stop, NULL);
	elapsed = GET_MS(start, stop);
	printf("elapsed time: %.04f ms.\n", elapsed);

	if ((length = hasError(y, length)) >= 0)
		printf("Wrong value at index %d: %.02f\n", length, y[length]);

	free(x);
	free(y);
	return (0);
}