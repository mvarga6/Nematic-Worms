
#ifndef __WORMS_KERNEL__CALCULATE_THETA_H__
#define __WORMS_KERNEL__CALCULATE_THETA_H__

#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

__global__ void CalculateThetaKernel(float *r,
									int rpitch,
									float *thphi,
									int tpitch)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < dev_Params._NPARTICLES)
	{
		int rshift = rpitch / sizeof(float);
		int tshift = tpitch / sizeof(float);

		int p = id % dev_Params._NP;
		if (p < (dev_Params._NP - 1))
		{
			int ix = id + 0 * rshift;
			int iy = id + 1 * rshift;
			int iz = id + 2 * rshift;

			int pp1 = id + 1;
			float dx = r[pp1]			 - r[ix];
			float dy = r[pp1 + rshift]	 - r[iy];
			float dz = r[pp1 + 2*rshift] - r[iz];
			DevicePBC(dx, dev_simParams._XBOX);
			DevicePBC(dy, dev_simParams._YBOX);
			float rmag = sqrtf(dx*dx + dy*dy + dz*dz);
			thphi[id] = atan2f(dy / rmag, dx / rmag);
			thphi[id + tshift] = PI / 2.0f - asinf(dz / rmag);
		}
	}
}

//.. launch one for each worm
__global__ void FinishCalculateThetaKernel(float *thphi, int tpitch){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < dev_Params._NWORMS){

		int tshift = tpitch / sizeof(float);

		// get to last particle in worm tid
		int id = (tid * dev_Params._NP) + (dev_Params._NP - 1);
		thphi[id] = thphi[id - 1];
		thphi[id + tshift] = thphi[(id - 1) + tshift];
	}
}

#endif