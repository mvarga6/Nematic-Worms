#ifndef __WORMS_KERNELS_H__
#define __WORMS_KERNELS_H__
// 2D
// -----------------
//.. cuda includes
// -----------------
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "math_functions.h"
#include <stdio.h>

// ------------------------------
//.. individual kernel includes
// ------------------------------
#include "NWWormsKernel_InternalForce.h"
#include "NWWormsKernel_Noise.h"
#include "NWWormsKernel_LennardJones.h"
#include "NWWormsKernel_DriveForce.h"
#include "NWWormsKernel_LandscapeForce.h"
#include "NWWormsKernel_CalculateTheta.h"
#include "NWWormsKernel_UpdateSystem.h"
#include "NWWormsKernel_SetNList.h"
#include "NWWormsKernel_BondBending.h"

// --------------------
// Just for debugging
// --------------------

__global__ void AddForce(float *f, int fpitch, int dim, float force){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){
		int fshift = fpitch / sizeof(float);
		int index = id + dim * fshift;
		f[index] += force;
	}
}

#endif