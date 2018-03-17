#pragma once

#include "../include_cuda.h"

namespace NW
{
	class GpuObject
	{
	public:
		__host__ virtual cudaError ToGpu() = 0;
		__host__ virtual cudaError FromGpu() = 0;
	};
}