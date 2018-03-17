#pragma once

#include "../../include_main.h"
#include "../typedefs.h"
#include "../../defs.h"

namespace NW
{
	class Physics
	{
	public:
		
		//
		// Returns a function for calculating lennard jones
		//
		__storage__ static real LennardJones(real rr, real epsilon, real sigmato6th);

	};
}

