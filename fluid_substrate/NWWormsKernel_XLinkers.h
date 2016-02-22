#ifndef __WORMS_KERNEL__XLINKERS_H__
#define __WORMS_KERNEL__XLINKERS_H__
// -----------------------------------------------------------------------------------------
#include "NWDeviceFunctions.h"
#include "NWParams.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"

// -----------------------------------------------------------------------------------------
//	Count the current number of cross-links (launch one)
__global__ void XLinkerCountKernel(int *xlink, int *xcount){
	int new_count = 0;
	for (int i = 0; i < dev_Params._NPARTICLES; i++)
		if (xlink[i] != -1) new_count++;
	(*xcount) = new_count;
}

// -----------------------------------------------------------------------------------------
// Randomly attempt to break some bonds
__global__ void XLinkerBreakKernel(int *xlink, float* randNum, float percentLoss)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){
		if (xlink[id] != -1){ // if linked
			if (randNum[id] < percentLoss){
				xlink[id] = -1;
			}
		}
	}
}

// -----------------------------------------------------------------------------------------
//	Update the cross linkers based on intended percentages
__global__ void XLinkerUpdateKernel(float *r,
							  int rshift,
							  int *xlink,
							  int *nlist,
							  int nlshift,
							  float *randNum, // one per particle needed uniform
							  float change_percent,
							  float linkCutoff)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){
		if (change_percent < 0) { // remove certain percentage
			const float target_percent = abs(change_percent);
			if (xlink[id] != -1) { // if linked
				if (randNum[id] < target_percent){ // attempt remove xlink
					xlink[id] = -1; // remove link
				}
			}
		}
		// FIX SO LOWER NUMBER NEIGHBORS AREN'T PICKED FIRST !!!!
		else { // add certain percentage 
			if (xlink[id] == -1) { // if unlinked
				if (randNum[id] < change_percent) { // if attempting adding 
					float rid[3], dr[3], _r[3], _rr;
					rid[0] = r[id];
					rid[1] = r[id + rshift];
					rid[2] = r[id + 2 * rshift];
					dr[0] = dr[1] = dr[2] = 0.0f;
					for (int n = 0; n < dev_Params._NMAX; n++){ // find someone to link with
						int nid = nlist[id + n * nlshift]; // neighbor id
						if (nid == -1) break; // end of neighbors
						_r[0] = r[nid];
						_r[1] = r[nid + rshift];
						_r[2] = r[nid + 2 * rshift];
						_rr = CalculateRR_3d(rid, _r, dr); // distane sqrd
						if (_rr > linkCutoff*linkCutoff) continue; // if close enough
						xlink[id] = nid; // assign linkage
					}
				}
			}
		}
	}
}
// ----------------------------------------------------------------------------------------
__global__ void XLinkerForceKernel(float *f,
								   int fshift,
								   float *r,
								   int rshift,
								   int *xlink,
								   bool f_on_self) 
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES){
		int xid = xlink[id]; // cross linked to
		if (xid != -1){ // if linked to someone
			float k = dev_Params._Kx;
			int pid;
			if (f_on_self) {
				pid = id; // apply force to self
				k = -k; // set force direction
			}
			else pid = xid; // apply force to particle linked to
			float rid[3], rnab[3], dr[3]; // local position vectors
			for (int d = 0; d < 3; d++){ // for all dimensions
				rid[d] = r[id + d*rshift]; // assign values for id
				rnab[d] = r[xid + d*rshift]; // assign values for xid
			}
			float _r = sqrt(CalculateRR_3d(rid, rnab, dr)); // distance
			float _f = -k * (_r - dev_Params._Lx) / _r; // magnitude of force
			for (int d = 0; d < 3; d++) // for all dimensions
				f[pid + d*fshift] += _f * dr[d]; // apply force component
		}
	}
} // --------------------------------------------------------------------------------------
#endif