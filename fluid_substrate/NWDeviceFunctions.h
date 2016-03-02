// Functions only for execution on GPU device
// 5.12.15
// Mike Varga
// 2D
#ifndef _DEVICE_FUNCTIONS_H
#define _DEVICE_FUNCTIONS_H
#include "NWmain.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"
//-----------------------------------------------------------------------------------
__device__ void DevicePBC(float &pos, float L)
{
	if (pos > L / 2.0f) pos -= L;
	if (pos < -L / 2.0f) pos += L;
}
//-----------------------------------------------------------------------------------
__device__ void DeviceMovementPBC(float &R, float L)
{
	if (R > L) R -= L;
	if (R < 0) R += L;
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
// ----------------------------------------------------------------------------------
__device__ float dot(const float v1[_D_], const float v2[_D_]){
	float result = 0.0f;
	for_D_ result += (v1[d] * v2[d]);
	return result;
}
// --------------------------------------------------------------------------------
__device__ float mag(const float v[_D_]){
	return sqrt(dot(v,v));
}
//-----------------------------------------------------------------------------------
__device__ float CalculateRR(const float _rid[_D_], const float _rnab[_D_], float _dr[_D_]){
	float _r_[_D_];
	for_D_ _r_[d] = _rnab[d] - _rid[d];
	//float _x = _rnab[0] - _rid[0];
	//float _y = _rnab[1] - _rid[1];
	//float _z = _rnab[2] - _rid[2];
	DevicePBC(_r_[0], dev_simParams._XBOX);
	DevicePBC(_r_[1], dev_simParams._YBOX);
	for_D_ _dr[d] = _r_[d];
	//_dr[0] = _x;
	//_dr[1] = _y;
	//_dr[2] = _z;
	//return (_dr[0]*_dr[0] + _dr[1]*_dr[1] + _dr[2]*_dr[2]);
	return dot(_r_, _r_);
}
//-----------------------------------------------------------------------------------
__device__ float CalculateLJ(const float _rr){
	return dev_Params._LJ_AMP*((dev_Params._2SIGMA6 / (_rr*_rr*_rr*_rr*_rr*_rr*_rr)) - (1.00000f / (_rr*_rr*_rr*_rr)));
}
//-----------------------------------------------------------------------------------
__device__ float CalculateLJ(const float _rr, const float sig, const float eps){
	const float _2sig6 = pow(sig, 6.0f);
	const float _lj_amp = 12.0f * eps * _2sig6;
	return _lj_amp*((_2sig6 / (_rr*_rr*_rr*_rr*_rr*_rr*_rr)) - (1.00000f / (_rr*_rr*_rr*_rr)));
}
//---------------------------------------------------------------------------------
#endif