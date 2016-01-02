
#ifndef __WORMS_KERNEL__LANDSCAPE_FORCE_H__
#define __WORMS_KERNEL__LANDSCAPE_FORCE_H__

#include "NWmain.h"
#include "NWParams.h"
#include "NWSimulationParameters.h"
#include "NWWormsParameters.h"
//old
/*__global__ void WormsLandscapeKernel(float *fx, float *fy, float *x, float *y){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){
		
		float _KK = 2.0f * PI * _N / dev_simParams._XBOX;

		float dx = x[id] - (dev_simParams._XBOX / 2.0f);
		float dy = y[id] - (dev_simParams._YBOX / 2.0f);
		DevicePBC(dx, dev_simParams._XBOX);
		DevicePBC(dy, dev_simParams._YBOX);
		float r = sqrtf(dx * dx + dy * dy);
		float coskr = cosf(_KK * r);
		fx[id] += -_KK * (coskr / r) * dx;
		fy[id] += -_KK * (coskr / r) * dy;
	}

}*/

__global__ void WormsLandscapeKernel(float *f,
									 int fpitch,
									 float *r,
									 int rpitch){

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){

		int fshift = fpitch / sizeof(float);
		int rshift = rpitch / sizeof(float);

		//.. harmonic potential zeroed around z = 0
		f[id + 2 * fshift] -= dev_Params._LANDSCALE * r[id + 2 * rshift];
	}
}

#endif