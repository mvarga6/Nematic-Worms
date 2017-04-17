#ifndef __WORMS_KERNEL__DRIVE_FORCE_H__
#define __WORMS_KERNEL__DRIVE_FORCE_H__
// 2D
// ----------------------------------------------------------------------------------------
#include "NWmain.h"
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
// -----------------------------------------------------------------------------------------
__global__ void DriveForceKernel(float *f, int fshift,
								 float *r, int rshift,
								 bool * alive){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (!dev_Params._EXTENSILE){
		if (id < dev_Params._NPARTICLES){
			const int w = id / dev_Params._NP; // worm id
			const int p = id % dev_Params._NP; // particle in chain
			if ((alive[w]) && (p < dev_Params._NP - 2)){ // if not head
				float u[_D_], dr[_D_], rid[_D_], rnab[_D_], umag;
				for_D_ rid[d] = r[id + d*rshift]; // get pos of id particle
				for_D_ rnab[d] = r[(id + 2) + d*rshift]; // get pos of next in chain
				umag = sqrt(CalculateRR(rid, rnab, dr)); // calculate displacement vector and mag
				for_D_ u[d] = dr[d] / umag; // make unit vector
				if (_D_ == 2) Rotate2D(u, dev_Params._DRIVE_ROT); // works for 2d only
				for_D_ f[(id + 1) + d*fshift] += dev_Params._DRIVE * u[d]; // apply drive along unit vector
			}
		}
	}
}

#endif