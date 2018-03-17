#pragma once

#include "../include_cuda.h"
#include "typedefs.h"
#include "../defs.h"

#include "SimulationContainer.cuh"

namespace NW
{
	class PeriodicSquareBox :
		 public SimulationContainer
	{
	public:

		__storage__ PeriodicSquareBox();

		__storage__ ~PeriodicSquareBox();

		real Lx, Ly, Lz;
		real L[_D_];

		__storage__ BoundaryFunction GetBoundaryFunction();
		__storage__ BoundaryForce GetBoundaryForce();

	private:
		//
		// Provide the functions to return
		//
		__storage__ void boundaryFunction(real r[_D_]);
		__storage__ void boundaryForce(real r[_D_], real f[_D_]);
	};
}
