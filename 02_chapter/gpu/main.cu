#include <stdio.h>
#include <math.h>

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

int
main(int argc, char* argv[])
{
	int length = 1 << 20;	// 1 048 576
	float* x = (float*)malloc(length * sizeof(float));
	float* y = (float*)malloc(length * sizeof(float));

	initArray(x, length, 1.0f);
	initArray(x, length, 2.0f);

	add(x, y, length);

	return (0);
}