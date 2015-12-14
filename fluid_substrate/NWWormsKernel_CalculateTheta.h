
#ifndef __WORMS_KERNEL__CALCULATE_THETA_H__
#define __WORMS_KERNEL__CALCULATE_THETA_H__

#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

__global__ void CalculateThetaKernel(float *x, float *y, float *z, float *theta, float *phi)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < dev_Params._NPARTICLES)
	{
		int p = id % dev_Params._NP;
		if (p < (dev_Params._NP - 1))
		{
			int pp1 = id + 1;
			float dx = x[pp1] - x[id];
			float dy = y[pp1] - y[id];
			float dz = z[pp1] - z[pp1];
			DevicePBC(dx, dev_simParams._XBOX);
			DevicePBC(dy, dev_simParams._YBOX);
			float rmag = sqrtf(dx*dx + dy*dy + dz*dz);
			theta[id] = atan2f(dy / rmag, dx / rmag);
			phi[id] = PI / 2.0f - asinf(dz / rmag);
		}
		/*else
		{
			theta[id] = theta[id - 1];
		}*/
	}
}

//.. launch one for each worm
__global__ void FinishCalculateThetaKernel(float *theta, float *phi){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < dev_Params._NWORMS){

		// get to last particle in worm tid
		int id = (tid * dev_Params._NP) + (dev_Params._NP - 1);
		theta[id] = theta[id - 1];
		phi[id] = phi[id - 1];
	}
}

#endif