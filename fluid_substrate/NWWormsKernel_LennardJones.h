
#ifndef __WORMS_KERNEL__LENNARD_JONES_H__
#define __WORMS_KERNEL__LENNARD_JONES_H__

#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

__global__ void LennardJonesNListKernel(float *f,
										int fshift,
										float *r,
										int rshift,
										int *nlist,
										int nshift )
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES)
	{
		//int fshift = fpitch / sizeof(float);
		//int rshift = rpitch / sizeof(float);
		//int nshift = nlpitch / sizeof(int);

		int ix = id;
		int iy = id + rshift;
		int iz = id + 2 * rshift;

		float fid[3], rid[3], dr[3], _r[3];
		float _f, _rr;
		fid[0] = 0.0f; fid[1] = 0.0f; fid[2] = 0.0f;
		rid[0] = r[ix];
		rid[1] = r[iy];
		rid[2] = r[iz];

		//.. loop through neighbors
		for (int i = 0; i < dev_Params._NMAX; i++)
		{
			//.. neighbor id
			int nid = nlist[id + i * nshift];

			//.. if no more players
			if (nid == -1) break;

			_r[0] = r[nid];
			_r[1] = r[nid + rshift];
			_r[2] = r[nid + 2 * rshift];

			_rr = CalculateRR_3d(rid, _r, dr);
			//float dx = x[nid] - x[id];
			//float dy = y[nid] - y[id];
			//DevicePBC(dx, dev_simParams._XBOX);
			//DevicePBC(dy, dev_simParams._YBOX);
			//float rr = dx * dx + dy * dy;

			//printf("nnid = %i\trr = %f", nid, _rr);

			//.. stop if too far
			if (_rr > dev_Params._R2CUT) continue;

			//.. calculate LJ force
			_f = CalculateLJ_3d(_rr);

			//.. apply forces to directions
			for (int d = 0; d < 3; d++)
				fid[d] -= _f * dr[d];
		}

		//.. assign tmp to memory
		f[id] += fid[0];
		f[id + fshift] += fid[1];
		f[id + 2*fshift] += fid[2];

#ifdef __PRINT_FORCES__
		if (id == __PRINT_INDEX__)
			printf("\n\tLJ Kernel:\n\tf = { %f, %f, %f }", f[id], f[id + fshift], f[id + 2*fshift]);
#endif
	}
}

#endif