#include "io.h"


void ShowError(cudaError_t Status)
{
	if (cudaSuccess != Status)
	{
		fprintf(stderr, "\nfail msg:  '%s'\n", cudaGetErrorString(Status));
		abort();
	}
}