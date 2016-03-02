
#ifndef __WORMS_KERNEL__UPDATE_SYSTEM_H__
#define __WORMS_KERNEL__UPDATE_SYSTEM_H__
// 2D
#include "NWmain.h"
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

//.. Update positions and velocities of particles then save forces
__global__ void UpdateSystemKernel(float *f,
								   int fshift,
								   float *f_old,
								   int foshift,
								   float *v,
								   int vshift, 
								   float *r,
								   int rshift,
								   float dt)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES)
	{
		//.. local components
		float dv[_D_], dr[_D_], rid[_D_];
		for_D_ rid[d] = r[id + d*rshift];

		//.. change in velocity
		for_D_ dv[d] = 0.5f * (f[id + d*fshift] + f_old[id + d*foshift]) * dt;

		//.. change in position
		for_D_ dr[d] = v[id + d*vshift] * dt + 0.5f * f_old[id + d*foshift] * dt * dt;

		//.. save forces
		for_D_ f_old[id + d*foshift] = f[id + d*fshift];

		//.. update positions
		for_D_ rid[d] += dr[d];

		//.. boundary conditions
		AdjPosPBC(rid, dev_simParams._BOX);

		for_D_ r[id + d*rshift] = rid[d];

		//.. update velocities
		for_D_ v[id + d*vshift] += dv[d];
		//DeviceMovementPBC(r[id], dev_simParams._XBOX);
		//DeviceMovementPBC(r[id + rshift], dev_simParams._YBOX);
	}
}


// ---------------------------------------------------------------------------------------
__global__ void FastUpdateKernel(float *f, int fshift,
								 float *f_old, int foshift,
								 float *v, int vshift,
								 float *r, int rshift,
								 float dt){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int bid = blockIdx.y;
	if (tid < dev_Params._NPARTICLES){

		const int fid = tid + bid * fshift;
		const int foid = tid + bid * foshift;
		const int vid = tid + bid * vshift;
		const int rid = tid + bid * rshift;

		float dvx = 0.5f * (f[fid] + f_old[foid]) * dt;
		float dx = v[vid] * dt + 0.5f * f_old[foid] * dt * dt;
		f_old[foid] = f[fid];
		r[rid] += dx;
		v[vid] += dvx;

		//.. for X and Y dimensions
		if (bid == 0) DeviceMovementPBC(r[rid], dev_simParams._XBOX);
		else if (bid == 1) DeviceMovementPBC(r[rid], dev_simParams._YBOX);

		f[fid] = 0.0f;
	}
}

#endif