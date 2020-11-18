#include "io.h"
#include <stdlib.h>
#include <stdio.h>

void ShowError(cudaError_t Status)
{
	if (cudaSuccess != Status)
	{
		fprintf(stderr, "\nfail msg:  '%s'\n", cudaGetErrorString(Status));
		abort();
	}
}