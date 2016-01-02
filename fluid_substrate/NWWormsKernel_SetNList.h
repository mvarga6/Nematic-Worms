
#ifndef __WORMS_KERNEL__SET_N_LIST_H__
#define __WORMS_KERNEL__SET_N_LIST_H__

#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

__global__ void SetNeighborList_N2Kernel(float *r,
										 int rpitch, 
										 int *nlist,
										 int nlpitch)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES)
	{
		//.. grab parameters (revert back to non-local variables)
		//__shared__ int np = dev_Params._NP;
		//__shared__ int nparts = dev_Params._NPARTICLES;
		//__shared__ int nmax = dev_Params._NMAX;
		//__shared__ float r2min = dev_Params._R2MIN;
		//__shared__ float r2cut = dev_Params._R2CUT;
		//__shared__ float buffer = dev_Params._BUFFER;

		int rshift = rpitch / sizeof(float);
		int nshift = nlpitch / sizeof(int);

		int n1 = id % dev_Params._NP;
		int w1 = id / dev_Params._NP;
		int found = 0;

		int ix = id;
		int iy = id + rshift;
		int iz = id + 2*rshift;

		float dr[3], _r[3], rid[3];
		rid[0] = r[ix]; 
		rid[1] = r[iy]; 
		rid[2] = r[iz];
		float _rr;
		for (int p2 = 0; p2 < dev_Params._NPARTICLES; p2++)
		{
			int w2 = p2 / dev_Params._NP;

			//.. if in the same worm
			float cutoff;
			int sep = 10; //.. any number 6 or greater would work here
			if (w2 == w1)
			{
				cutoff = dev_Params._R2MIN + dev_Params._BUFFER;
				int n2 = p2 % dev_Params._NP;
				sep = abs(n2 - n1);
			}
			else //.. normal cutoff
			{
				cutoff = dev_Params._R2CUT + dev_Params._BUFFER;
			}

			//.. skip if to near in same worm
			if (sep <= 5) continue;

			//.. add to nlist if within range
			_r[0] = r[p2]; 
			_r[1] = r[p2 + rshift]; 
			_r[2] = r[p2 + 2*rshift];
			_rr = CalculateRR_3d(rid, _r, dr);

			//dx = x[p2] - x[id];
			//dy = y[p2] - y[id];
			//DevicePBC(dx, dev_simParams._XBOX);
			//DevicePBC(dy, dev_simParams._YBOX);
			//rr = dx * dx + dy * dy;
			if ((_rr <= cutoff) && (found < dev_Params._NMAX))
			{
				//printf("\n%i on %i", p2, id);

				int plcid = id + found++ * nshift;
				nlist[plcid] = p2;
			}
		}
	}
}

#endif