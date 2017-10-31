#pragma once

#include "../../include_main.h"

namespace NW
{
	class PhysicsKernels
	{
	public:

		//
		// Abstact base class for physics models
		//
		virtual ~PhysicsKernels() = 0;

		//
		// Internal Forces - forces proesses inside bodies
		//
		virtual void InternalForces(real *f, int fshift, real *v, int vshift, real *r, int rshift, float *randNum, float noiseScaler) = 0;

		//
		// Thermal Forces - noise
		//
		virtual void ThermalForces(real *f, int fshift, float *rn, float mag) = 0;

		//
		// Particel-Particles Forces - between neighboring particles
		//
		virtual void ParticleParticleForces(real *f, int fshift, real *r, int rshift, int *nlist, int nshift) = 0;

		//
		// Active forces - driving force etc
		//
		virtual void ActiveForces(real *f, int fshift, real *r, int rshift, bool *alive, int *dir) = 0;

		//
		// External Forces - forces due to interactionsn with environment
		//
		virtual void ExternalForces(real *f, int fshift, real *r, int rshift) = 0;

		//
		// Container Forces
		//
		virtual void ContainerForces(real *f, int fshift, real *r, int rshift) = 0;

		//
		// Bond Bending Forces - forces from bending bodies
		//
		virtual void BondBendingForces(real *f, int fshift, real* r, int rshift) = 0;

		//
		// Update System - moves particles based on forces
		//
		virtual void UpdateSystem(real *f, int fshift, real *f_old, int foshift, real *v, int vshift, real *r, int rshift, int *cell, int cshift, real dt) = 0;

		//
		// Update System - moves particles based on forces
		//
		virtual void UpdateSystemFast(real *f, int fshift, real *f_old, int foshift, real *v, int vshift, real *r, int rshift, int *cell, int cshift, real dt) = 0;
	};
}

