#pragma once

#include "../include_main.h"
#include "ParticleProperties.h"

namespace NW
{
	class ParticleContainer
	{
	
	public:

		ParticleContainer();
		ParticleContainer();
		~ParticleContainer();

		real *f, // forces
			 *v, // velocities
			 *r; // positions

		// optional properties can be set
		ParticleProperties *props = NULL;
	};
}

