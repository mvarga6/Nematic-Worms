#pragma once

#include "../../include_main.h"
#include "../KernelLaunchInfo.h"
#include "PhysicsKernels.h"

namespace NW
{
	class HoegerPhysics :
		public PhysicsKernels
	{
		// how to launch the theads
		KernelLaunchInfo *launchInfo;

	public:

		__host__
		HoegerPhysics(KernelLaunchInfo *kernelLaunchInfo);
		~HoegerPhysics();

		//
		// Internal Forces - forces proesses inside bodies
		//
		virtual void InternalForces(real *f, int fshift, real *v, int vshift, real *r, int rshift, float *randNum, float noiseScaler);		

		//
		// Thermal Forces - noise
		//
		virtual void ThermalForces(real *f, int fshift, float *rn, float mag);		

		//
		// Particel-Particles Forces - between neighboring particles
		//
		virtual void ParticleParticleForces(real *f, int fshift, real *r, int rshift, int *nlist, int nshift);	

		//
		// Active forces - driving force etc
		//
		virtual void ActiveForces(real *f, int fshift, real *r, int rshift, bool *alive, int *dir);

		//
		// External Forces - forces due to interactionsn with environment
		//
		virtual void ExternalForces(real *f, int fshift, real *r, int rshift);

		//
		// Container Forces
		//
		virtual void ContainerForces(real * f, int fshift, real * r, int rshift);

		//
		// Bond Bending Forces - forces from bending bodies
		//
		virtual void BondBendingForces(real *f, int fshift, real* r, int rshift);	

		//
		// Update System - moves particles based on forces
		//
		virtual void UpdateSystem(real *f, int fshift, real *f_old, int foshift, real *v, int vshift, real *r, int rshift, int *cell, int cshift, real dt);
		
		//
		// Update System - moves particles based on forces
		//
		virtual void UpdateSystemFast(real *f, int fshift, real *f_old, int foshift, real *v, int vshift, real *r, int rshift, int *cell, int cshift, real dt);	
		
	private:

		//
		// Private implementation of the kernels underlying the physics methods
		//
		__global__ static void __InternalForces(real *f, int fshift, real *v, int vshift, real *r, int rshift, float *randNum, float noiseScaler);
		__global__ static void __ThermalForces(real *f, int fshift, float *rn, float mag);
		__global__ static void __ParticleParticleForces(real *f, int fshift, real *r, int rshift, int *nlist, int nshift);
		__global__ static void __ActiveForces(real *f, int fshift, real *r, int rshift, bool *alive, int *dir);
		__global__ static void __ExternalForces(real *f, int fshift, real *r, int rshift);
		__global__ static void __ContainerForces(real * f, int fshift, real * r, int rshift);
		__global__ static void __BondBendingForces(real *f, int fshift, real* r, int rshift);
		__global__ static void __UpdateSystem(real *f, int fshift, real *f_old, int foshift, real *v, int vshift, real *r, int rshift, int *cell, int cshift, real dt);
		__global__ static void __UpdateSystemFast(real *f, int fshift, real *f_old, int foshift, real *v, int vshift, real *r, int rshift, int *cell, int cshift, real dt);
	};
}

