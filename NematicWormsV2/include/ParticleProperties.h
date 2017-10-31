#pragma once

#include "../include_main.h"

namespace NW
{
	//
	// Class that lets properties be defined
	// on the particles to be used in the calculates
	//
	class ParticleProperties
	{
	public:
		ParticleProperties();
		~ParticleProperties();

		// The properties (max 6) - these need to be
		// setup by the Physics model
		real *P1, *P2, *P3, *P4, *P5, *P6;
	};
}

