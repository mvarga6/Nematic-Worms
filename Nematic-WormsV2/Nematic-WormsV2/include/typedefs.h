#pragma once

#include "../defs.h"

namespace NW
{
	// boundary functions
	typedef void(*BoundaryFunction)(real r[_D_]);
	typedef void(*BoundaryForce)(real r[_D_], real f[_D_]);

	typedef void(*SingleDimensionBoundaryFunction)(real &x);
	typedef void(*SingleDimensionBoundaryForce)(real &x, real &fx);

	// Metric functions
	typedef void(*DisplacementMetric)(real d[_D_]);
	typedef real(*DistanceSquaredMetric)(real r1[_D_], real r2[_D_], real dr[_D_]);

	// Force function types
	typedef real(*InteractionFunction)(real in);
}