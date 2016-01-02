#ifndef __WORMS_KERNEL__DRIVE_FORCE_H__
#define __WORMS_KERNEL__DRIVE_FORCE_H__
// ----------------------------------------------------------------------------------------
#include "NWmain.h"
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
// -----------------------------------------------------------------------------------------
__global__ void DriveForceKernel(float *f,
								 int fpitch,
								 float *thphi,
								 int tpitch){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){

		int fshift = fpitch / sizeof(float);
		int tshift = tpitch / sizeof(float);

		//float phi = atan2f(z[id], dev_Params._L1);
		float sinphi = sinf(thphi[id + tshift]);
		f[id]			 += dev_Params._DRIVE * cosf(thphi[id]) * sinphi;
		f[id + fshift]	 += dev_Params._DRIVE * sinf(thphi[id]) * sinphi;
		f[id + 2*fshift] += dev_Params._DRIVE * cosf(thphi[id + tshift]);

#ifdef __PRINT_FORCES__
		if (id == __PRINT_INDEX__)
			printf("\n\tDrive Kernel:\n\tf = { %f, %f, %f }", f[id], f[id + fshift], f[id + 2*fshift]);
#endif
	}
}

#endif