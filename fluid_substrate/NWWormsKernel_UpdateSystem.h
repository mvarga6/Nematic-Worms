
#ifndef __WORMS_KERNEL__UPDATE_SYSTEM_H__
#define __WORMS_KERNEL__UPDATE_SYSTEM_H__

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
		//int fshift = fpitch / sizeof(float);
		//int foshift = fopitch / sizeof(float);
		//int vshift = vpitch / sizeof(float);
		//int rshift = rpitch / sizeof(float);
	
		//.. change in velocity
		float dvx = 0.5f * (f[id] + f_old[id]) * dt;
		float dvy = 0.5f * (f[id + fshift] + f_old[id + foshift]) * dt;
		float dvz = 0.5f * (f[id + 2*fshift] + f_old[id + 2*foshift]) * dt;

		//.. change in position
		float dx = v[id] * dt + 0.5f * f_old[id] * dt * dt;
		float dy = v[id + vshift] * dt + 0.5f * f_old[id + foshift] * dt * dt;
		float dz = v[id + 2*vshift] * dt + 0.5f * f_old[id + 2*foshift] * dt * dt;

		//.. save forces
		f_old[id] = f[id];
		f_old[id + foshift] = f[id + fshift];
		f_old[id + 2*foshift] = f[id + 2*foshift];

		//.. update positions and velocities
		r[id] += dx;
		r[id + rshift] += dy;
		r[id + 2*rshift] += dz;
		v[id] += dvx;
		v[id + vshift] += dvy;
		v[id + 2*vshift] += dvz;

		//.. boundary conditions
		DeviceMovementPBC(r[id], dev_simParams._XBOX);
		DeviceMovementPBC(r[id + rshift], dev_simParams._YBOX);
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

		//int fshift = fpitch / sizeof(float);
		//int foshift = fopitch / sizeof(float);
		//int vshift = vpitch / sizeof(float);
		//int rshift = rpitch / sizeof(float);

		int fid = tid + bid * fshift;
		int foid = tid + bid * foshift;
		int vid = tid + bid * vshift;
		int rid = tid + bid * rshift;

		float dvx = 0.5f * (f[fid] + f_old[foid]) * dt;
		float dx = v[vid] * dt + 0.5f * f_old[foid] * dt * dt;
		f_old[foid] = f[fid];
		r[rid] += dx;
		v[vid] += dvx;

		if (bid == 0) DeviceMovementPBC(r[rid], dev_simParams._XBOX);
		else if (bid == 1) DeviceMovementPBC(r[rid], dev_simParams._YBOX);

		f[fid] = 0.0f;
	}
}

#endif