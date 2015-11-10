
#ifndef __WORMS_KERNEL__LENNARD_JONES_H__
#define __WORMS_KERNEL__LENNARD_JONES_H__

#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

__global__ void LennardJonesNListKernel(float *fx, float *fy, float *x, float *y, int *nlist)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES)
	{
		int listid = id * dev_Params._NMAX;
		float fxid = 0.0;
		float fyid = 0.0;

		//.. loop through neighbors
		for (int i = 0; i < dev_Params._NMAX; i++)
		{
			//.. neighbor id
			int nid = nlist[listid + i];

			//.. if no more players
			if (nid == -1) break;

			float dx = x[nid] - x[id];
			float dy = y[nid] - y[id];
			DevicePBC(dx, dev_simParams._XBOX);
			DevicePBC(dy, dev_simParams._YBOX);
			float rr = dx * dx + dy * dy;

			//.. stop if too far
			if (rr > dev_Params._R2CUT) continue;

			//.. calculate LJ force
			float ff = dev_Params._LJ_AMP*((dev_Params._2SIGMA6 / (rr*rr*rr*rr*rr*rr*rr)) - (1.00000f / (rr*rr*rr*rr)));
			fxid -= ff * dx;
			fyid -= ff * dy;
		}

		//.. assign tmp to memory
		fx[id] += fxid;
		fy[id] += fyid;

#ifdef __PRINT_FORCES__
		if (id == 0)	
			printf("\n\tLJ Kernel:\n\tfx = %f,\tfy = %f\n", fx[id], fy[id]);
#endif
	}
}

#endif