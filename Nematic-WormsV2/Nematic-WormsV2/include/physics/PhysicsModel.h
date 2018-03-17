#pragma once

#include "../../include_main.h"
#include "../Particles.h"
#include "../Filaments.h"
#include "../Environment.h"
#include "../RandomNumbersService.h"
#include "../NeighborsGraph.h"
#include "../../defs.h"

namespace NW
{
	class PhysicsModel
	{


	public:

		//
		// Abstact base class for physics models
		//
		virtual ~PhysicsModel() = 0;

		//
		// Internal Forces - forces proesses inside bodies
		//
		virtual void InternalForces(Particles*, Filaments *) = 0;

		//
		// Thermal Forces - noise
		//
		virtual void ThermalForces(Particles*, Filaments*, float kBT) = 0;

		//
		// Particel-Particles Forces - between neighboring particles
		//
		virtual void ParticleParticleForces(Particles*, Filaments*, NeighborsGraph*) = 0;

		//
		// Active forces - driving force etc
		//
		virtual void ActiveForces(Particles*, Filaments*) = 0;

		//
		// External Forces - forces due to interactionsn with environment
		//
		virtual void ExternalForces(Particles*, Filaments*, Environment*) = 0;

		//
		// Container Forces
		//
		virtual void ContainerForces(Particles*, Filaments*) = 0;

		//
		// Bond Bending Forces - forces from bending bodies
		//
		virtual void BondBendingForces(Particles*, Filaments*) = 0;

		//
		// Update System - moves particles based on forces
		//
		virtual void UpdateSystem(Particles*, real dt) = 0;

		//
		// Update System - moves particles based on forces
		//
		//virtual void UpdateSystemFast(Particles*, Environment *,real dt) = 0;
	};
}

