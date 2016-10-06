#ifndef __WORMS_KERNEL__INTERNAL_FORCE_H__
#define __WORMS_KERNEL__INTERNAL_FORCE_H__
// 2D
// ------------------------------------------------------------------------------------------
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

// ------------------------------------------------------------------------------------------
__global__ void InterForceKernel(float *f,
								 int fshift,
								 float *v,
								 int vshift,
								 float *r,
								 int rshift,
								 float *randNum,
								 float noiseScaler)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){

		//.. particel number in worm
		int p = id % dev_Params._NP;
		int w = id / dev_Params._NP;

		//.. local memory
		float rid[_D_], fid[_D_], vid[_D_];

		//.. init local as needed
		for_D_{
			rid[d] = r[id + d*rshift];
			fid[d] = 0.0f;
			vid[d] = v[id + d*vshift];
		}
			
		//.. 1st neighbor spring forces ahead
		if (p < (dev_Params._NP - 1))
		{
			int pp1 = id + 1;
			float rnab[_D_], dr[_D_];
			float _r, _f;
			for_D_ rnab[d] = r[pp1 + d*rshift];
			_r = sqrt(CalculateRR(rid, rnab, dr));
			_f = -(dev_Params._K1 * (_r - dev_Params._L1)) / _r;
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. 1st neighbor spring forces behind
		if (p > 0)
		{
			int pm1 = id - 1;
			float rnab[_D_], dr[_D_];
			float _r, _f;
			for_D_ rnab[d] = r[pm1 + d*rshift];
			_r = sqrt(CalculateRR(rid, rnab, dr));
			_f = -(dev_Params._K1 * (_r - dev_Params._L1)) / _r;
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. LJ between inter-worm particles
		for (int id2 = w*dev_Params._NP; id2 < (w + 1)*dev_Params._NP; id2++)
		{
			int p2 = id2 % dev_Params._NP;
			int sep = abs(p2 - p);
			if (sep <= 2) continue; //.. ignore 1st and 2nd neighbors
			float rnab[_D_], dr[_D_];
			float rr, _f;
			for_D_ rnab[d] = r[id2 + d*rshift];
			rr = CalculateRR(rid, rnab, dr);
			if (rr > dev_Params._R2MIN) continue; //.. repulsive only
			_f = CalculateLJ(rr);
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. viscous drag
		for_D_ fid[d] -= dev_Params._GAMMA * vid[d];

		//.. thermal fluctuations
		for_D_ fid[d] += noiseScaler * randNum[id + d*dev_Params._NPARTICLES];

		//.. assign temp fxid and fyid to memory
		for_D_ f[id + d*fshift] += fid[d];
	}
}

#endif
