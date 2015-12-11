
#ifndef __WORMS_KERNEL__DRIVE_FORCE_H__
#define __WORMS_KERNEL__DRIVE_FORCE_H__

#include "NWmain.h"
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"

__global__ void DriveForceKernel(float *fx, float *fy, float *fz,
								float *z, float *theta){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){

		float phi = atan2f(z[id], dev_Params._L1);
		float cosphi = cosf(phi);
		fx[id] += dev_Params._DRIVE * cosf(theta[id]) * cosphi;
		fy[id] += dev_Params._DRIVE * sinf(theta[id]) * cosphi;
		fz[id] += dev_Params._DRIVE * sinf(phi);

#ifdef __PRINT_FORCES__
		if (id == 0)
			printf("\n\tDrive Kernel:\n\tfx = %f,\tfy = %f\n", fx[id], fy[id]);
#endif
	}
}

#endif