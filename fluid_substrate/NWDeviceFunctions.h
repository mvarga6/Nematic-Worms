// Functions only for execution on GPU device
// 5.12.15
// Mike Varga

#ifndef _DEVICE_FUNCTIONS_H
#define _DEVICE_FUNCTIONS_H
#include "NWmain.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"
//-----------------------------------------------------------------------------------
__device__ void DevicePBC(float &dr, float L)
{
	if (dr > L / 2.0f) dr -= L;
	if (dr < -L / 2.0f) dr += L;
}
//-----------------------------------------------------------------------------------
__device__ void DeviceMovementPBC(float &pos, float L)
{
	if (pos > L) pos -= L;
	if (pos < 0) pos += L;
}
//-----------------------------------------------------------------------------------
__device__ bool InList(int arg, int* list, int listSize)
{
	for (int i = 0; i < listSize; i++)
	{
		if (arg == list[i]) return true;
	}
	return false;
}
//-----------------------------------------------------------------------------------
__device__ float CalculateRR_3d(const float rid[3], const float rnab[3], float dr[3]){
	dr[0] = rnab[0] - rid[0];
	dr[1] = rnab[1] - rid[1];
	dr[2] = rnab[2] - rid[2];
	DevicePBC(dr[0], dev_simParams._XBOX);
	DevicePBC(dr[1], dev_simParams._YBOX);
	return (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
}
//-----------------------------------------------------------------------------------
__device__ float CalculateLJ_3d(float &rr){
	return dev_Params._LJ_AMP*((dev_Params._2SIGMA6 / (rr*rr*rr*rr*rr*rr*rr)) - (1.00000f / (rr*rr*rr*rr)));
}
// ----------------------------------------------------------------------------------
#endif