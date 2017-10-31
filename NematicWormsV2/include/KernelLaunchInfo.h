#pragma once

#include "../include_main.h"

namespace NW
{
	class KernelLaunchInfo
	{
		// only particles container can make 
		KernelLaunchInfo();

	public:
		
		~KernelLaunchInfo();

		dim3 ParticleGridStructure;
		dim3 ParticleBlockStructure;
		dim3 BodiesGridStructure;
		dim3 BodiesBlockStructure;
	};
}

