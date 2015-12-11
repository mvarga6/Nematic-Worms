
#ifndef __WORMS_KERNEL__LENNARD_JONES_H__
#define __WORMS_KERNEL__LENNARD_JONES_H__

#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

__global__ void LennardJonesNListKernel(float *fx, float *fy, float *fz,
										float *x, float *y, float *z,
										int *nlist)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES)
	{
		int listid = id * dev_Params._NMAX;
		float fid[3], dr[3], r[3], rid[3];
		float _f, _rr;
		fid[0] = fid[1] = fid[2] = 0.0f;
		rid[0] = x[id]; rid[1] = y[id]; rid[2] = z[id];

		//.. loop through neighbors
		for (int i = 0; i < dev_Params._NMAX; i++)
		{
			//.. neighbor id
			int nid = nlist[listid + i];

			//.. if no more players
			if (nid == -1) break;

			r[0] = x[nid]; r[1] = y[nid]; r[2] = z[nid];
			_rr = CalculateRR_3d(rid, r, dr);
			//float dx = x[nid] - x[id];
			//float dy = y[nid] - y[id];
			//DevicePBC(dx, dev_simParams._XBOX);
			//DevicePBC(dy, dev_simParams._YBOX);
			//float rr = dx * dx + dy * dy;

			//.. stop if too far
			if (_rr > dev_Params._R2CUT) continue;
			
			//.. calculate LJ force
			_f = CalculateLJ_3d(_rr);

			for (int d = 0; d < 3; d++)
				fid[d] -= _f * dr[d];
		}

		//.. assign tmp to memory
		fx[id] += fid[0];
		fy[id] += fid[1];
		fz[id] += fid[2];

#ifdef __PRINT_FORCES__
		if (id == 0)	
			printf("\n\tLJ Kernel:\n\tfx = %f,\tfy = %f\n", fx[id], fy[id]);
#endif
	}
}

#endif