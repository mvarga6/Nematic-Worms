#pragma once

#include "../../include_main.h"
#include "../../defs.h"
#include "../typedefs.h"

namespace NW
{
	//
// Prototypes of the kernels underlying the physics methods
//
	__global__ void HoegerInternalForces(real *f, real *v, real *r, int N, int np, real kbond, real lbond, DistanceSquaredMetric RR, InteractionFunction interaction);
	__global__ void HoegerThermalForces(real *f, int N, float * rn, float mag);
	__global__ void HoegerParticleParticleForces(real *f, real *r, int N, int np, int *nlist, int nmax, DistanceSquaredMetric RR, InteractionFunction interaction);
	__global__ void HoegerActiveForces(real *f, real *r, int N, int np, real *drive, DistanceSquaredMetric RR);
	__global__ void HoegerExternalForces(real *f, real *v, int N, real gamma);
	__global__ void HoegerContainerForces(real *f, real *r, int N, BoundaryForce container_interaction);
	__global__ void HoegerBondBendingForces(real *f, real *r, int N, int nworms, int np, real kbend, DisplacementMetric disp_metric);
	__global__ void HoegerUpdateSystem(real *f, real *f_old, real *v, real *r, int N, BoundaryFunction bc, real dt);
	//__global__ static void __UpdateSystemFast(real *f, real *f_old, real* v, real * r, int N, BoundaryFunction bc, real dt); 
}