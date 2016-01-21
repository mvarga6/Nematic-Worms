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
__device__ float CalculateRR_3d(const float _rid[3], const float _rnab[3], float _dr[3]){
	float _x = _rnab[0] - _rid[0];
	float _y = _rnab[1] - _rid[1];
	float _z = _rnab[2] - _rid[2];
	DevicePBC(_x, dev_simParams._XBOX);
	DevicePBC(_y, dev_simParams._YBOX);
	_dr[0] = _x;
	_dr[1] = _y;
	_dr[2] = _z;
	return (_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2]);
}
//-----------------------------------------------------------------------------------
__device__ float CalculateLJ_3d(const float _rr){
	return dev_Params._LJ_AMP*((dev_Params._2SIGMA6 / (_rr*_rr*_rr*_rr*_rr*_rr*_rr)) - (1.00000f / (_rr*_rr*_rr*_rr)));
}
// ----------------------------------------------------------------------------------
__device__ float dot(const float v1[3], const float v2[3]){
	float result = 0.0f;
	for (int j = 0; j < 3; j++)
		result += (v1[j] * v2[j]);
	return result;
}
// --------------------------------------------------------------------------------
__device__ float mag(const float v[3]){
	return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
//---------------------------------------------------------------------------------
#endif