#pragma once

#include "../defs.h"

namespace NW
{
	class FilamentProperties
	{
	public:

		// bond stretching energy
		real Kbond, Lbond;

		// bending energy
		real Kbend;

		// Particles per filament
		int Np;
	};
}

