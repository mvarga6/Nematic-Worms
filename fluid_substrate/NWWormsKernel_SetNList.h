
#ifndef __WORMS_KERNEL__SET_N_LIST_H__
#define __WORMS_KERNEL__SET_N_LIST_H__
// 2D
//----------------------------------------------------------------------------------------------
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"
//----------------------------------------------------------------------------------------------
__global__ void SetNeighborList_N2Kernel(float *r,
										 int rshift, 
										 int *nlist,
										 int nlshift)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){

		//const int n1 = id % dev_Params._NP;
		const int w1 = id / dev_Params._NP;
		int found = 0;

		float dr[_D_], _r[_D_], rid[_D_];
		for_D_ rid[d] = r[id + d*rshift];
		//rid[0] = r[ix]; 
		//rid[1] = r[iy]; 
		//rid[2] = r[iz];
		float _rr;
		const float cutoff = dev_Params._R2CUT + dev_Params._BUFFER;
		for (int p2 = 0; p2 < dev_Params._NPARTICLES; p2++)
		{
			int w2 = p2 / dev_Params._NP;

			//.. stop if in same worm
			if (w2 == w1) continue;

			//.. add to nlist if within range
			for_D_ _r[d] = r[p2 + d*rshift];
			//_r[0] = r[p2]; 
			//_r[1] = r[p2 + rshift]; 
			//_r[2] = r[p2 + 2*rshift];
			_rr = CalculateRR(rid, _r, dr);
			
			if ((_rr <= cutoff) && (found < dev_Params._NMAX))
			{
				int plcid = id + found++ * nlshift;
				nlist[plcid] = p2;

				//if (found == dev_Params._NMAX)
				//	printf("\nFilled nlist for particle %i", id);
			}
		}
	}
}
// -------------------------------------------------------------------------------------------
__global__ void FastSetNeighborListKernel(float *r, int rshift,
	int nlist, int nlshift){
	int id = threadIdx.x + blockDim.x * blockIdx.x; // particle index
	int nid = threadIdx.y + blockDim.y * blockIdx.y; // target index
	int nparts = dev_Params._NPARTICLES;
	int np = dev_Params._NP;
	if (id < nparts && nid < nparts){
		int w1 = id / np;
		int w2 = nid / np;
		if (w1 != w2){
			//float rid[3], rnab[3];
			//.. unfinshed
		}


	}
}
#endif