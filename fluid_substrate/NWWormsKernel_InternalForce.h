#ifndef __WORMS_KERNEL__INTERNAL_FORCE_H__
#define __WORMS_KERNEL__INTERNAL_FORCE_H__
// ------------------------------------------------------------------------------------------
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"
//#define __PRINT_FORCES__
//#define __PRINT_POS_1__
//#define __PRINT_POS_2__
//#define __PRINT_INDEX__ 999
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

#ifdef __PRINT_SHIFTS__
		if (id == __PRINT_INDEX__) 
			printf("\nf,v,r shifts: %i ,\t%i ,\t%i", fshift, vshift, rshift);
#endif
		//.. particel number in worm
		int p = id % dev_Params._NP;
		int w = id / dev_Params._NP;

		//.. local memory
		float rid[3], fid[3], vid[3]; // , dr[3], rnab[3];
		//float _r, _f;

		//.. init local as needed
		for (int i = 0; i < 3; i++){
			rid[i] = r[id + i * rshift];
			//rnab[i] = 0.0f;
			//dr[i] = 0.0f;
			fid[i] = 0.0f;
			vid[i] = v[id + i * vshift];
		}

#ifdef __PRINT_POS_1__
		if (id == __PRINT_INDEX__) 
			printf("\nrid = { %f, %f, %f }", rid[0], rid[1], rid[2]);
#endif
			
		//.. first neighbor forces
		//.. 1st neighbor spring forces ahead
		if (p < (dev_Params._NP - 1))
		{
			int pp1 = id + 1;
			float rnab[3];
			float dr[3];
			float _r, _f;
			rnab[0] = r[pp1 + 0 * rshift];
			rnab[1] = r[pp1 + 1 * rshift];
			rnab[2] = r[pp1 + 2 * rshift];

#ifdef __PRINT_POS_2__
			if (id == __PRINT_INDEX__) 
				printf("\nrnab = { %f, %f, %f }", rnab[0], rnab[1], rnab[2]);
#endif

			_r = sqrt(CalculateRR_3d(rid, rnab, dr));
			_f = -(dev_Params._K1 * (_r - dev_Params._L1)) / _r;
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
			float rnab[3];
			float dr[3];
			float _r, _f;
			rnab[0] = r[pm1 + 0 * rshift];
			rnab[1] = r[pm1 + 1 * rshift];
			rnab[2] = r[pm1 + 2 * rshift];
			_r = sqrt(CalculateRR_3d(rid, rnab, dr));
			_f = -(dev_Params._K1 * (_r - dev_Params._L1)) / _r;
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
//		if (p < (dev_Params._NP - 2))
//		{
//			int pp2 = id + 2;
//			float rnab[3];
//			float dr[3];
//			float _r, _f;
//			rnab[0] = r[pp2 + 0 * rshift];
//			rnab[1] = r[pp2 + 1 * rshift];
//			rnab[2] = r[pp2 + 2 * rshift];
//			_r = sqrtf(CalculateRR_3d(rid, rnab, dr));
//			_f = -(dev_Params._K2 * (_r - dev_Params._L2)) / _r;
//			for (int d = 0; d < 3; d++)
//				fid[d] -= _f * dr[d];
//			//float dx = x[pp2] - xid;
//			//float dy = y[pp2] - yid;
//			//DevicePBC(dx, dev_simParams._XBOX);
//			//DevicePBC(dy, dev_simParams._YBOX);
//			//float rr = dx*dx + dy*dy;
//			//float _r = sqrtf(rr);
//			//float ff = -dev_Params._K2 * (_r - dev_Params._L2) / _r;
//			//float ffx = ff * dx;
//			//float ffy = ff * dy;
//			//fxid -= ffx;
//			//fyid -= ffy;
////#ifdef _DAMPING1
////			float dvx = vx[pp2] - vxid;
////			float dvy = vy[pp2] - vyid;
////			fxid += dev_Params._DAMP * dvx;
////			fyid += dev_Params._DAMP * dvy;
////#endif
//		}
//
//		//.. 2nd neighbor spring forces behind
//		if (p > 1)
//		{
//			int pm2 = id - 2;
//			float rnab[3];
//			float dr[3];
//			float _r, _f;
//			rnab[0] = r[pm2 + 0 * rshift];
//			rnab[1] = r[pm2 + 1 * rshift];
//			rnab[2] = r[pm2 + 2 * rshift];
//			_r = sqrtf(CalculateRR_3d(rid, rnab, dr));
//			_f = -(dev_Params._K2 * (_r - dev_Params._L2)) / _r;
//			for (int d = 0; d < 3; d++)
//				fid[d] -= _f * dr[d];
//			//float dx = x[pm2] - xid;
//			//float dy = y[pm2] - yid;
//			//DevicePBC(dx, dev_simParams._XBOX);
//			//DevicePBC(dy, dev_simParams._YBOX);
//			//float rr = dx*dx + dy*dy;
//			//float _r = sqrtf(rr);
//			//float ff = -dev_Params._K2 * (_r - dev_Params._L2) / _r;
//			//float ffx = ff * dx;
//			//float ffy = ff * dy;
//			//fxid -= ffx;
//			//fyid -= ffy;
////#ifdef _DAMPING1
//			//float dvx = vx[pm2] - vxid;
//			//float dvy = vy[pm2] - vyid;
//			//fxid += dev_Params._DAMP * dvx;
//			//fyid += dev_Params._DAMP * dvy;
////#endif
//		}

		//.. 3nd neighbor spring forces ahead
//		if (p < (dev_Params._NP - 3))
//		{
//			int pp3 = id + 3;
//			float rnab[3];
//			float dr[3];
//			float _r, _f;
//			rnab[0] = r[pp3 + 0 * rshift];
//			rnab[1] = r[pp3 + 1 * rshift];
//			rnab[2] = r[pp3 + 2 * rshift];
//			_r = sqrtf(CalculateRR_3d(rid, rnab, dr));
//			_f = -(dev_Params._K3 * (_r - dev_Params._L3)) / _r;
//			for (int d = 0; d < 3; d++)
//				fid[d] -= _f * dr[d];
//			//float dx = x[pp3] - xid;
//			//float dy = y[pp3] - yid;
//			//DevicePBC(dx, dev_simParams._XBOX);
//			//DevicePBC(dy, dev_simParams._YBOX);
//			//float rr = dx*dx + dy*dy;
//			//float _r = sqrtf(rr);
//			//float ff = -dev_Params._K3 * (_r - dev_Params._L3) / _r;
//			//float ffx = ff * dx;
//			//float ffy = ff * dy;
//			//fxid -= ffx;
//			//fyid -= ffy;
//
////#ifdef _DAMPING2
////			float dvx = vx[pp3] - vxid;
////			float dvy = vy[pp3] - vyid;
////			fxid += dev_Params._DAMP * dvx;
////			fyid += dev_Params._DAMP * dvy;
////#endif
//		}
//
//		//.. 3nd neighbor spring forces behind
//		if (p > 2)
//		{
//			int pm3 = id - 3;
//			float rnab[3];
//			float dr[3];
//			float _r, _f;
//			rnab[0] = r[pm3 + 0 * rshift];
//			rnab[1] = r[pm3 + 1 * rshift];
//			rnab[2] = r[pm3 + 2 * rshift];
//			_r = sqrtf(CalculateRR_3d(rid, rnab, dr));
//			_f = -(dev_Params._K3 * (_r - dev_Params._L3)) / _r;
//			for (int d = 0; d < 3; d++)
//				fid[d] -= _f * dr[d];
//			//float dx = x[pm3] - xid;
//			//float dy = y[pm3] - yid;
//			//DevicePBC(dx, dev_simParams._XBOX);
//			//DevicePBC(dy, dev_simParams._YBOX);
//			//float rr = dx*dx + dy*dy;
//			//float _r = sqrtf(rr);
//			//float ff = -dev_Params._K3 * (_r - dev_Params._L3) / _r;
//			//float ffx = ff * dx;
//			//float ffy = ff * dy;
//			//fxid -= ffx;
//			//fyid -= ffy;
//
////#ifdef _DAMPING2
////			float dvx = vx[pm3] - vxid;
////			float dvy = vy[pm3] - vyid;
////			fxid += dev_Params._DAMP * dvx;
////			fyid += dev_Params._DAMP * dvy;
////#endif
//		}

		//.. 4nd neighbor spring forces ahead
		/*if (p < (_NP - 4))
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
		for (int id2 = w*dev_Params._NP; id2 < (w + 1)*dev_Params._NP; id2++)
		{

			int p2 = id2 % dev_Params._NP;
			int sep = abs(p2 - p);
			
			//.. everything but 1st neighbors
			if (sep <= 1) continue;

			float rnab[3];
			float dr[3];
			float rr, _f;
			rnab[0] = r[id2];
			rnab[1] = r[id2 + rshift];
			rnab[2] = r[id2 + 2 * rshift];
			rr = CalculateRR_3d(rid, rnab, dr);

			//.. repulsive only
			if (rr > dev_Params._R2MIN) continue;

			_f = CalculateLJ_3d(rr);
			for (int d = 0; d < 3; d++)
				fid[d] -= _f * dr[d];
 
			//if (abs(id2 - id) >= 5)
			//{
			//	float dx = x[id2] - xid;
			//	float dy = y[id2] - yid;
			//	DevicePBC(dx, _XBOX);
			//	DevicePBC(dy, _YBOX);
			//	float rr = dx*dx + dy*dy;
			//	//.. only repulsive for intraworm
			//	if (rr <= _R2MIN)
			//	{
			//		float ff = _LJ_AMP*((_2SIGMA6 / (rr*rr*rr*rr*rr*rr*rr)) - (1.0f / (rr*rr*rr*rr)));
			//		fxid -= ff * dx;
			//		fyid -= ff * dy;
			//	}
			//}
		}

		//.. viscous drag
		for (int d = 0; d < 3; d++)
			fid[d] -= dev_Params._GAMMA * vid[d];

		//.. thermal fluctuations
		for (int d = 0; d < 3; d++)
			fid[d] += noiseScaler * randNum[id + d*dev_Params._NPARTICLES];

		//fxid += scalor * randNum[id];
		//fyid += scalor * randNum[id + dev_Params._NPARTICLES];
		//printf("tid = %i:\tR = { %f, %f }\n", id, randNum[id], randNum[id + dev_Params._NPARTICLES]);

		//.. assign temp fxid and fyid to memory
		f[id + 0 * fshift] += fid[0];
		f[id + 1 * fshift] += fid[1];
		f[id + 2 * fshift] += fid[2];


#ifdef __PRINT_FORCES__
		if (id == __PRINT_INDEX__) 
			printf("\n\tInternal Kernel:\n\tf = { %f, %f, %f }", f[id], f[id + fshift], f[id + 2*fshift]);
#endif
	}
}

#endif
