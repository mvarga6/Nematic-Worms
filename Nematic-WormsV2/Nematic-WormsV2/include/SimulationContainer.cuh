#pragma once


#include "GpuObject.h"
#include "../include_cuda.h"

namespace NW
{
	class SimulationContainer
	{
	public:
		__storage__ virtual BoundaryFunction GetBoundaryFunction() = 0;
		__storage__ virtual BoundaryForce GetBoundaryForce() = 0;
	};
}