#pragma once

#include "Parameters.h"
#include "Particles.h"
#include "physics/PhysicsModel.h"
#include "Environment.h"

namespace NW
{
	class SimulationController
	{
	public:

		//
		// Injected Controller with elements it needs to run
		//
		SimulationController(PhysicsModel *model);
		~SimulationController();
	};
}

