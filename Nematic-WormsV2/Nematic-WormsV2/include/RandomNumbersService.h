#pragma once

#include "../include_main.h"
#include "../defs.h"

namespace NW
{
	/* ----------------------------------------------------------------------------
	*	brief: Gaussian Random Number Generator
	*
	*	Allows calls to one static random number generator.  Has one fixed size
	*	ptr on the device that fills when calls and returns that ptr.  Used in
	*	other classes when random numbers are needed.  This way there are not lots
	*	of different number generators and device ptrs full of random numbers.
	------------------------------------------------------------------------------*/
	class RandomNumbersService {
		//.. device ptr for numbers
		real * dev_n;
		//curandState *dev_s;

		//.. actual number generator
		curandGenerator_t * generator;

		//.. status of the generator
		std::vector<curandStatus_t> status;
		std::vector<cudaError_t> errorState;

		//.. size of allocation on device
		const size_t alloc_size;
		int max_size;
		int Threads_Per_Block;
		int Blocks_Per_Kernel;

		//.. Gaussian distribution specs
		const float mean;
		const float stddev;

	public:
		RandomNumbersService(int maxCallSize, float distributionMean, float standardDeviation);
		~RandomNumbersService();

		real* Get(unsigned count, bool uniform);
		void DisplayErrors();

	private:
		void AllocateGPUMemory(int N);
		void FreeGPUMemory();
		void CheckStatus(curandStatus_t stat);
		void CheckSuccess(cudaError_t err);
	};
}

