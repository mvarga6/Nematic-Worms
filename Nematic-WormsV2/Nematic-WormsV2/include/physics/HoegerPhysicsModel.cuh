#pragma once

#include "../../include_main.h"
#include "../../include/FilamentProperties.h"
#include "../KernelLaunchInfo.h"
#include "PhysicsModel.h"
#include "../SystemMetrics.h"
#include "../Filaments.h"
#include "../SquareBox.h"
#include "../../defs.h"
#include "../BoundaryConditionService.h"
#include "../../include/Environment.h"

namespace NW
{
	class HoegerModelParameters
	{
	public:
		real RatioAlive, ReverseRate, Drive;
	};

	class HoegerPhysicsModel :
		public PhysicsModel
	{
		HoegerModelParameters * parameters;
		SystemMetrics * metrics;
		BoundaryConditionService * boundary;
		KernelLaunchInfo * launchInfo;

		// interaction functions
		InteractionFunction intrafilament_interaction;
		InteractionFunction particle_particle_interaction;

		// service for random numbers
		RandomNumbersService* random;

	public:

		HoegerPhysicsModel(HoegerModelParameters * parameters, SystemMetrics *metrics, BoundaryConditionService *boundaryService);
		~HoegerPhysicsModel();

		//
		// Internal Forces - forces proesses inside bodies
		//
		virtual void InternalForces(Particles*, Filaments*);

		//
		// Thermal Forces - noise
		//
		virtual void ThermalForces(Particles*, Filaments*, float kBT);

		//
		// Particel-Particles Forces - between neighboring particles
		//
		virtual void ParticleParticleForces(Particles*, Filaments*, NeighborsGraph*);

		//
		// Active forces - driving force etc
		//
		virtual void ActiveForces(Particles*, Filaments*);

		//
		// External Forces - forces due to interactionsn with environment
		//
		virtual void ExternalForces(Particles*, Filaments*, Environment*);

		//
		// Container Forces
		//
		virtual void ContainerForces(Particles*, Filaments *);

		//
		// Bond Bending Forces - forces from bending bodies
		//
		virtual void BondBendingForces(Particles*, Filaments *);

		//
		// Update System - moves particles based on forces
		//
		virtual void UpdateSystem(Particles*, real dt);

		//
		// Update System - moves particles based on forces
		//
		//virtual void UpdateSystemFast(Particles*, Environment *, real dt);

	};

	
}

