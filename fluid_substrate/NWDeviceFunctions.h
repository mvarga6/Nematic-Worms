// Functions only for execution on GPU device
// 5.12.15
// Mike Varga

#ifndef _DEVICE_FUNCTIONS_H
#define _DEVICE_FUNCTIONS_H
#include "NWmain.h"

__device__ void DevicePBC(float &dr, float L)
{
	if (dr > L / 2.0) dr -= L;
	if (dr < -L / 2.0) dr += L;
}

__device__ void DeviceMovementPBC(float &pos, float L)
{
	if (pos > L) pos -= L;
	if (pos < 0) pos += L;
}

__device__ bool InList(int arg, int* list, int listSize)
{
	for (int i = 0; i < listSize; i++)
	{
		if (arg == list[i]) return true;
	}
	return false;
}

#endif