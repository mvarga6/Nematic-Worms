#pragma once

#include "../include_main.h"

namespace NW
{
	class KernelLaunchInfo
	{
		

	public: 
		KernelLaunchInfo();
		~KernelLaunchInfo();

		dim3 ParticlesGridStructure;
		dim3 ParticlesBlockStructure;
		dim3 BodiesGridStructure;
		dim3 BodiesBlockStructure;
	};
}

