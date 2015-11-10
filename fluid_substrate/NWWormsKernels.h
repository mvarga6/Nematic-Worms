
#ifndef __WORMS_KERNELS_H__
#define __WORMS_KERNELS_H__

// -------------------------------------------
//.. cuda includes
// -------------------------------------------
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "math_functions.h"
#include <stdio.h>

// -------------------------------------------
//.. individual kernel includes and prototypes
// -------------------------------------------
__global__ void InterForceKernel(float *fx, float *fy, float *vx, float *vy, float *x, float *y, float * randNum);
#include "NWWormsKernel_InternalForce.h"

__global__ void LennardJonesNListKernel(float *fx, float *fy, float *x, float *y, int *nlist);
#include "NWWormsKernel_LennardJones.h"

__global__ void DriveForceKernel(float *fx, float *fy, float *theta);
#include "NWWormsKernel_DriveForce.h"

__global__ void WormsLandscapeKernel(float *fx, float *fy, float *x, float *y);
#include "NWWormsKernel_LandscapeForce.h"

__global__ void CalculateThetaKernel(float *x, float *y, float *theta);
#include "NWWormsKernel_CalculateTheta.h"

__global__ void UpdateSystemKernel(float *fx, float *fy, float *fx_old, float *fy_old, float *vx, float *vy, float *x, float *y);
#include "NWWormsKernel_UpdateSystem.h"

__global__ void SetNeighborList_N2Kernel(float *x, float *y, int *nlist);
#include "NWWormsKernel_SetNList.h"

__global__ void SetXListWormsKernel(float *wx, float *wy, float *flx, float *fly, int *xlist);
#include "NWWormsKernel_SetXList.h"

#endif