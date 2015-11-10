
#ifndef __ERROR_HANDLER_H__
#define __ERROR_HANDLER_H__

#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>

void ErrorHandler(cudaError_t Status)
{
	if (cudaSuccess != Status)
	{
		fprintf(stderr, "\nfail msg:  '%s'\n", cudaGetErrorString(Status));
		//fprintf(stderr, "\npress any key to clear memory...");
		//cin.get();
		//ShutDownGPUDevice();
		//CleanUpHost();
		abort();
	}
}

#endif