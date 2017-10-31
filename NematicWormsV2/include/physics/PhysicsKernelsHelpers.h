#pragma once

#include "../../include_main.h"

namespace NW 
{
	class PhysicsKernelsHelpers
	{
	public:

		//
		// Abstract base class for methods that assist
		// the main Physics Kernels
		//
		virtual ~PhysicsKernelsHelpers() = 0;

		//
		// Assign Neighbors
		//
		virtual void FindNeighbors(real *r, int rshift, int *nlist, int nlshift, int *cell, int cshift) = 0;

		//
		// Post Update Method
		//
		virtual void PostUpdateCallback(real *r, int rshfit, real *v, int vshift, real *p1, int p1shift, real *p2, int p2shift) = 0;
	
	
		
	};
}