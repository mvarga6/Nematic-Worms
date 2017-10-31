#pragma once

#include "Parameters.h"
#include "ParticleContainer.h"
#include "physics/PhysicsKernels.h"

namespace NW
{
	class SimulationController
	{
	public:

		//
		// Injected Controller with elements it needs to run
		//
		SimulationController(ParticleContainer *particlesContainer, PhysicsKernels *physicsModel, Parameters *paramaters);
		~SimulationController();
	};
}

