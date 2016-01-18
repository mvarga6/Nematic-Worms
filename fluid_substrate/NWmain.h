// Includes all necessary files
// 5.12.15
// Mike Varga

// CUDA Library
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

// Other Includes
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include <time.h>
#include <ctime>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <assert.h>

// Host Function Prototypes
__host__ int ProcessCommandLine(int argc, char *argv[]);
//__host__ int OpenDataFiles(void);
__host__ void ErrorHandler(cudaError_t Status);
__host__ void MovementBC(float &, float L);
__host__ void PBC(float &, float L);

__host__ void ClockWorks(int &,std::clock_t &, std::clock_t &, std::clock_t &);
__host__ void PrintXYZ(float *wx, float *wy, float *flx, float *fly, int *xlist);
__host__ void SaveSimulationConfiguration(void);

