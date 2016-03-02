#ifndef __WORMS_KERNEL__DRIVE_FORCE_H__
#define __WORMS_KERNEL__DRIVE_FORCE_H__
// 2D
// ----------------------------------------------------------------------------------------
#include "NWmain.h"
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
// -----------------------------------------------------------------------------------------
__global__ void DriveForceKernel(float *f,
								 int fshift,
								 float *thphi,
								 int tshift){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){

		//float phi = atan2f(z[id], dev_Params._L1);
		const float sinphi = sinf(thphi[id + tshift]);
		const float u[3] = {
			cosf(thphi[id]) * sinphi,
			sinf(thphi[id]) * sinphi,
			cosf(thphi[id + tshift])
		};

		for_D_ f[id + d*fshift] += dev_Params._DRIVE * u[d];
		//f[id]			 += dev_Params._DRIVE * cosf(thphi[id]) * sinphi;
		//f[id + fshift]	 += dev_Params._DRIVE * sinf(thphi[id]) * sinphi;
		//f[id + 2*fshift] += dev_Params._DRIVE * cosf(thphi[id + tshift]);

#ifdef __PRINT_FORCES__
		if (id == __PRINT_INDEX__)
			printf("\n\tDrive Kernel:\n\tf = { %f, %f, %f }", f[id], f[id + fshift], f[id + 2*fshift]);
#endif
	}
}

#endif