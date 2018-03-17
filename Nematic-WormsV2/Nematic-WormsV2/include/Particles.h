#pragma once

#include "../include_main.h"
#include "../defs.h"
#include "ParticleProperties.h"

namespace NW
{
	class Particles
	{
	
	public:

		Particles(int Nparticle);
		~Particles();

		real *f, // forces
			 *f_old, // old forces
			 *v, // velocities
			 *r; // positions

		int N, shift;

		// optional properties can be set
		ParticleProperties *Props = NULL;
	};
}

