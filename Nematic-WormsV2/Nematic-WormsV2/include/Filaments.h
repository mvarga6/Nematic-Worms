#pragma once

#include "../defs.h"
#include "FilamentProperties.h"

namespace NW
{
	class Filaments
	{
	public:

		Filaments(int Nfilaments);
		~Filaments();

		int N;
		FilamentProperties *GlobalProperties;

		int *HeadIdx, // idx of head of each filament in particles list
			*Length, // lengths of each filament
			*Dir;

		real *Activity, // activity of the filament
			*Kbend, // bending scalers
			*Kbond; // stretching scaler
	};
}

