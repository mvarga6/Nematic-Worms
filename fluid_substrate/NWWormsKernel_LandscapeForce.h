
#ifndef __WORMS_KERNEL__LANDSCAPE_FORCE_H__
#define __WORMS_KERNEL__LANDSCAPE_FORCE_H__
// 2D
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
									 int fshift,
									 float *r,
									 int rshift){

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){

		const float eps = 20.0f; // potential depth
		const float sig = 60.0f; // attractor size
		float rid[_D_], dr[_D_], R_attr[_D_]; // position vectors
		for_D_ R_attr[d] = dev_simParams._BOX[d] / 2.0f; // center of dims
		for_D_ rid[d] = r[id + d*rshift]; // local particle position
		const float rr = CalculateRR(rid, R_attr, dr); // r^2
		const float _f = CalculateLJ(rr, sig, eps); // force calc
		for_D_ f[id + d * fshift] -= _f * dr[d]; // apply forces
	}
}

#endif