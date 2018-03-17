#pragma once

#include "../defs.h"
#include "typedefs.h"
#include "../include_cuda.h"

namespace NW
{
	enum class BoundaryConditionType : int
	{
		Periodic = 1,
		Open = 2,
		SoftWalls = 3,
		HardWalls = 4
	};

	enum class BoundaryShape : int
	{
		Square = 1,
		Circular = 2,
		Trapozoidal = 3
	};

	class BoundaryConditions
	{
	public:
		__storage__ BoundaryConditions() {};

		real Specs[_D_]; // size of area
		BoundaryConditionType Type[_D_]; // type of boundaries
		BoundaryShape Shape; // shape of boundary
		real Kwall; // parameter of walls
	};

	// namespace for hidden global types and objects
	namespace Internal
	{
		// the object used for bc info on the gpu
		__constant__ BoundaryConditions BC;
	}

	// object to pass into methods that interact with boundary
	class BoundaryConditionService
	{
		BoundaryConditions boundaryConditions;

		// the functions that get set based on creation
		BoundaryFunction boundaryFunction;
		BoundaryForce boundaryForce;

		SingleDimensionBoundaryFunction boundaryFunctions[_D_];
		SingleDimensionBoundaryForce boundaryForces[_D_];


	public:
		__storage__ BoundaryConditionService(BoundaryConditions boundaryConditions);

		__storage__ BoundaryFunction GetBoundaryFunction();
		__storage__ BoundaryForce GetBoundaryForce();

	private:
		__storage__ void SquareBC(real r[_D_]);
		__storage__ void CircularBC(real r[_D_]);
		__storage__ void TrapBC(real r[_D_]);

		__storage__ void SquareBF(real r[_D_], real f[_D_]);
		__storage__ void CircularBF(real r[_D_], real f[_D_]);
		__storage__ void TrapBF(real r[_D_], real f[_D_]);
	};
}