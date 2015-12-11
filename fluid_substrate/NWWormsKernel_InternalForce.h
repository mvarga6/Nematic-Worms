
#ifndef __WORMS_KERNEL__INTERNAL_FORCE_H__
#define __WORMS_KERNEL__INTERNAL_FORCE_H__

#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

__global__ void InterForceKernel(float *fx, float *fy, float *fz,
								 float *vx, float *vy, float *vz,
								 float *x, float *y, float *z, 
								 float * randNum)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){

		int p = id % dev_Params._NP;

		//.. local memory
		float rid[3], fid[3], dr[3], r[3];
		float _r, _f;
		rid[0] = x[id];
		rid[1] = y[id];
		rid[2] = z[id];
		fid[0] = fid[1] = fid[2] = 0.0f;
		dr[0] = dr[1] = dr[2] = 0.0f;
		r[0] = r[1] = r[2] = 0.0f;
#ifdef _DAMPING
		float vid[3];
		vid[0] = vx[id];
		vid[1] = vy[id];
		vid[2] = vz[id];
#endif

		//.. 1st neighbor spring forces ahead
		if (p < (dev_Params._NP - 1))
		{
			int pp1 = id + 1;
			r[0] = x[pp1]; r[1] = y[pp1]; r[2] = z[pp1];
			_r = sqrtf(CalculateRR_3d(rid, r, dr));
			_f = -dev_Params._K1 * (_r - dev_Params._L1) / _r;
			for (int d = 0; d < 3; d++)
				fid[d] -= _f * dr[d];
			//float dx = x[pp1] - xid;
			//float dy = y[pp1] - yid;
			//DevicePBC(dx, dev_simParams._XBOX);
			//DevicePBC(dy, dev_simParams._YBOX);
			//float rr = dx*dx + dy*dy;
			//float _r = sqrtf(rr);
			//float ffx = ff * dx;
			//float ffy = ff * dy;
			//fxid -= ffx;
			//fyid -= ffy;
		}

		//.. 1st neighbor spring forces behind
		if (p > 0)
		{
			int pm1 = id - 1;
			r[0] = x[pm1]; r[1] = y[pm1]; r[2] = z[pm1];
			_r = sqrtf(CalculateRR_3d(rid, r, dr));
			_f = -dev_Params._K1 * (_r - dev_Params._L1) / _r;
			for (int d = 0; d < 3; d++)
				fid[d] -= _f * dr[d];
			//float dx = x[pm1] - xid;
			//float dy = y[pm1] - yid;
			//DevicePBC(dx, dev_simParams._XBOX);
			//DevicePBC(dy, dev_simParams._YBOX);
			//float rr = dx*dx + dy*dy;
			//float _r = sqrtf(rr);
			//float ff = -dev_Params._K1 * (_r - dev_Params._L1) / _r;
			//float ffx = ff * dx;
			//float ffy = ff * dy;
			//fxid -= ffx;
			//fyid -= ffy;
		}

		//.. 2nd neighbor spring forces ahead
		if (p < (dev_Params._NP - 2))
		{
			int pp2 = id + 2;
			r[0] = x[pp2]; r[1] = y[pp2]; r[2] = z[pp2];
			_r = sqrtf(CalculateRR_3d(rid, r, dr));
			_f = -dev_Params._K2 * (_r - dev_Params._L2) / _r;
			for (int d = 0; d < 3; d++)
				fid[d] -= _f * dr[d];
			//float dx = x[pp2] - xid;
			//float dy = y[pp2] - yid;
			//DevicePBC(dx, dev_simParams._XBOX);
			//DevicePBC(dy, dev_simParams._YBOX);
			//float rr = dx*dx + dy*dy;
			//float _r = sqrtf(rr);
			//float ff = -dev_Params._K2 * (_r - dev_Params._L2) / _r;
			//float ffx = ff * dx;
			//float ffy = ff * dy;
			//fxid -= ffx;
			//fyid -= ffy;

//#ifdef _DAMPING1
//			float dvx = vx[pp2] - vxid;
//			float dvy = vy[pp2] - vyid;
//			fxid += dev_Params._DAMP * dvx;
//			fyid += dev_Params._DAMP * dvy;
//#endif
		}

		//.. 2nd neighbor spring forces behind
		if (p > 1)
		{
			int pm2 = id - 2;
			r[0] = x[pm2]; r[1] = y[pm2]; r[2] = z[pm2];
			_r = sqrtf(CalculateRR_3d(rid, r, dr));
			_f = -dev_Params._K2 * (_r - dev_Params._L2) / _r;
			for (int d = 0; d < 3; d++)
				fid[d] -= _f * dr[d];
			//float dx = x[pm2] - xid;
			//float dy = y[pm2] - yid;
			//DevicePBC(dx, dev_simParams._XBOX);
			//DevicePBC(dy, dev_simParams._YBOX);
			//float rr = dx*dx + dy*dy;
			//float _r = sqrtf(rr);
			//float ff = -dev_Params._K2 * (_r - dev_Params._L2) / _r;
			//float ffx = ff * dx;
			//float ffy = ff * dy;
			//fxid -= ffx;
			//fyid -= ffy;

//#ifdef _DAMPING1
			//float dvx = vx[pm2] - vxid;
			//float dvy = vy[pm2] - vyid;
			//fxid += dev_Params._DAMP * dvx;
			//fyid += dev_Params._DAMP * dvy;
//#endif
		}

		//.. 3nd neighbor spring forces ahead
		if (p < (dev_Params._NP - 3))
		{
			int pp3 = id + 3;
			r[0] = x[pp3]; r[1] = y[pp3]; r[2] = z[pp3];
			_r = sqrtf(CalculateRR_3d(rid, r, dr));
			_f = -dev_Params._K3 * (_r - dev_Params._L3) / _r;
			for (int d = 0; d < 3; d++)
				fid[d] -= _f * dr[d];
			//float dx = x[pp3] - xid;
			//float dy = y[pp3] - yid;
			//DevicePBC(dx, dev_simParams._XBOX);
			//DevicePBC(dy, dev_simParams._YBOX);
			//float rr = dx*dx + dy*dy;
			//float _r = sqrtf(rr);
			//float ff = -dev_Params._K3 * (_r - dev_Params._L3) / _r;
			//float ffx = ff * dx;
			//float ffy = ff * dy;
			//fxid -= ffx;
			//fyid -= ffy;

//#ifdef _DAMPING2
//			float dvx = vx[pp3] - vxid;
//			float dvy = vy[pp3] - vyid;
//			fxid += dev_Params._DAMP * dvx;
//			fyid += dev_Params._DAMP * dvy;
//#endif
		}

		//.. 3nd neighbor spring forces behind
		if (p > 2)
		{
			int pm3 = id - 3;
			r[0] = x[pm3]; r[1] = y[pm3]; r[2] = z[pm3];
			_r = sqrtf(CalculateRR_3d(rid, r, dr));
			_f = -dev_Params._K3 * (_r - dev_Params._L3) / _r;
			for (int d = 0; d < 3; d++)
				fid[d] -= _f * dr[d];
			//float dx = x[pm3] - xid;
			//float dy = y[pm3] - yid;
			//DevicePBC(dx, dev_simParams._XBOX);
			//DevicePBC(dy, dev_simParams._YBOX);
			//float rr = dx*dx + dy*dy;
			//float _r = sqrtf(rr);
			//float ff = -dev_Params._K3 * (_r - dev_Params._L3) / _r;
			//float ffx = ff * dx;
			//float ffy = ff * dy;
			//fxid -= ffx;
			//fyid -= ffy;

//#ifdef _DAMPING2
//			float dvx = vx[pm3] - vxid;
//			float dvy = vy[pm3] - vyid;
//			fxid += dev_Params._DAMP * dvx;
//			fyid += dev_Params._DAMP * dvy;
//#endif
		}

		/*//.. 4nd neighbor spring forces ahead
		if (p < (_NP - 4))
		{
		int pp4 = id + 4;
		float dx = x[pp4] - xid;
		float dy = y[pp4] - yid;
		DevicePBC(dx, _XBOX);
		DevicePBC(dy, _YBOX);
		float rr = dx*dx + dy*dy;
		float _r = sqrtf(rr);
		float ff = -_K2 * 0.5f * (_r - _L2*2.0f) / _r;
		float ffx = ff * dx;
		float ffy = ff * dy;
		fxid -= ffx;
		fyid -= ffy;
		}

		//.. 4nd neighbor spring forces behind
		if (p > 3)
		{
		int pm4 = id - 4;
		float dx = x[pm4] - xid;
		float dy = y[pm4] - yid;
		DevicePBC(dx, _XBOX);
		DevicePBC(dy, _YBOX);
		float rr = dx*dx + dy*dy;
		float _r = sqrtf(rr);
		float ff = -_K2 * 0.5f * (_r - _L2 * 2.0f) / _r;
		float ffx = ff * dx;
		float ffy = ff * dy;
		fxid -= ffx;
		fyid -= ffy;
		}

		//.. 5nd neighbor spring forces ahead
		if (p < (_NP - 5))
		{
		int pp5 = id + 5;
		float dx = x[pp5] - xid;
		float dy = y[pp5] - yid;
		DevicePBC(dx, _XBOX);
		DevicePBC(dy, _YBOX);
		float rr = dx*dx + dy*dy;
		float _r = sqrtf(rr);
		float ff = -_K2 * 0.4f * (_r - _L1*5.0f) / _r;
		float ffx = ff * dx;
		float ffy = ff * dy;
		fxid -= ffx;
		fyid -= ffy;
		}

		//.. 5nd neighbor spring forces behind
		if (p > 4)
		{
		int pm5 = id - 5;
		float dx = x[pm5] - xid;
		float dy = y[pm5] - yid;
		DevicePBC(dx, _XBOX);
		DevicePBC(dy, _YBOX);
		float rr = dx*dx + dy*dy;
		float _r = sqrtf(rr);
		float ff = -_K2 * 0.4f * (_r - _L1 * 5.0f) / _r;
		float ffx = ff * dx;
		float ffy = ff * dy;
		fxid -= ffx;
		fyid -= ffy;
		}*/

		//.. LJ between inter-worm particles
		/*for (int id2 = w*_NP; id2 < w*_NP + _NP; id2++)
		{
		if (abs(id2 - id) >= 5)
		{
		float dx = x[id2] - xid;
		float dy = y[id2] - yid;
		DevicePBC(dx, _XBOX);
		DevicePBC(dy, _YBOX);
		float rr = dx*dx + dy*dy;
		//.. only repulsive for intraworm
		if (rr <= _R2MIN)
		{
		float ff = _LJ_AMP*((_2SIGMA6 / (rr*rr*rr*rr*rr*rr*rr)) - (1.0f / (rr*rr*rr*rr)));
		fxid -= ff * dx;
		fyid -= ff * dy;
		}
		}
		}*/

#ifdef _DRAG
		for (int d = 0; d < 3; d++)
			fid[d] -= dev_Params._GAMMA * vid[d];
#endif

#ifdef _NOISE
		float scalor = sqrtf(2.0f * dev_Params._GAMMA * dev_Params._KBT / dev_simParams._DT);
		for (int d = 0; d < 3; d++)
			fid[d] += scalor * randNum[id + d*dev_Params._NPARTICLES];
		//fxid += scalor * randNum[id];
		//fyid += scalor * randNum[id + dev_Params._NPARTICLES];
		//printf("tid = %i:\tR = { %f, %f }\n", id, randNum[id], randNum[id + dev_Params._NPARTICLES]);
#endif

		//.. assign temp fxid and fyid to memory
		fx[id] = fid[0];
		fy[id] = fid[1];
		fz[id] = fid[2];

#ifdef __PRINT_FORCES__
		if (id == 0)	
			printf("\n\tInternal Kernel:\n\tfx = %f,\tfy = %f\n", fxid, fyid);
#endif
	}
}

#endif