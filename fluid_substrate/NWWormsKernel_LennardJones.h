
#ifndef __WORMS_KERNEL__LENNARD_JONES_H__
#define __WORMS_KERNEL__LENNARD_JONES_H__
// 2D
//-------------------------------------------------------------------------------------------------
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"
//------------------------------------------------------------------------------------------------
__global__ void LennardJonesNListKernel(float *f,
										int fshift,
										float *r,
										int rshift,
										int *nlist,
										int nshift )
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){

		float fid[_D_], rid[_D_], dr[_D_], _r[_D_];
		float _f, _rr;
		for_D_{ fid[d] = 0.0f; rid[d] = r[id + d*rshift];}
		//fid[0] = 0.0f; fid[1] = 0.0f; fid[2] = 0.0f;
		//rid[0] = r[ix];
		//rid[1] = r[iy];
		//rid[2] = r[iz];

		//.. loop through neighbors
		for (int i = 0; i < dev_Params._NMAX; i++)
		{
			//.. neighbor id
			int nid = nlist[id + i*nshift];

			//.. if no more players
			if (nid == -1) break;

			for_D_ _r[d] = r[nid + d*rshift];
			_rr = CalculateRR(rid, _r, dr);

			//.. stop if too far
			if (_rr > dev_Params._R2CUT) continue;

			//.. calculate LJ force
			_f = CalculateLJ(_rr);

			//.. apply forces to directions
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. assign tmp to memory
		for_D_ f[id + d*fshift] += fid[d];

#ifdef __PRINT_FORCES__
		if (id == __PRINT_INDEX__)
			printf("\n\tLJ Kernel:\n\tf = { %f, %f, %f }", f[id], f[id + fshift], f[id + 2*fshift]);
#endif
	}
}
//---------------------------------------------------------------------------------------------------
__global__ void FastLennardJonesNListKernel(float *f, int fshift,
											float *r, int rshift,
											int *nlist, int nlshift){
	int pid = threadIdx.x + blockDim.x * blockIdx.x; // particle index
	int nab = threadIdx.y + blockDim.y * blockIdx.y; // neighbor count
	if ((pid < dev_Params._NPARTICLES) && (nab < dev_Params._NMAX)){
		int nid = nlist[pid + nab * nlshift];
		if (nid != -1){
			float rid[_D_], rnab[_D_], dr[_D_], _f, rr;
			for_D_ rid[d] = r[pid + d*rshift];
			for_D_ rnab[d] = r[nid + d*rshift];
			rr = CalculateRR(rid, rnab, dr);
			if (rr < dev_Params._R2CUT){
				_f = CalculateLJ(rr);
				for_D_ f[pid + d*fshift] -= _f * dr[d];
			}
		}
	}

}
#endif
