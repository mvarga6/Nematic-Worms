
#ifndef __RANDOM_NUMBER_GENERATOR_H__
#define __RANDOM_NUMBER_GENERATOR_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"
#include "NWmain.h"

/* ----------------------------------------------------------------------------
*	brief: Gaussian Random Number Generator
*
*	Allows calls to one static random number generator.  Has one fixed size
*	ptr on the device that fills when calls and returns that ptr.  Used in
*	other classes when random numbers are needed.  This way there are not lots
*	of different number generators and device ptrs full of random numbers.
------------------------------------------------------------------------------*/
class GRNG {
	//.. device ptr for numbers
	float * dev_r;

	//.. actual number generator
	curandGenerator_t * generator;

	//.. status of the generator
	std::vector<curandStatus_t> status;
	std::vector<cudaError_t> errorState;

	//.. size of allocation on device
	const size_t alloc_size;

	//.. Gaussian distribution specs
	const float mean;
	const float stddev;

public:
	GRNG(int maxCallSize, float distributionMean, float standardDeviation);
	~GRNG();

	float* Get(unsigned count);

	void DisplayErrors();

private:
	void AllocateGPUMemory();
	void FreeGPUMemory();
	void CheckStatus(curandStatus_t stat);
	void CheckSuccess(cudaError_t err);
};
// ---------------------------------------------------------------------------
//	PUBLIC METHODS
// ---------------------------------------------------------------------------
GRNG::GRNG(int maxCallSize, float distributionMean, float standardDeviation)
	: generator(new curandGenerator_t), dev_r(NULL),
	alloc_size(sizeof(float)*maxCallSize), mean(distributionMean),
	stddev(standardDeviation){
	CheckStatus(curandCreateGenerator(this->generator, CURAND_RNG_PSEUDO_DEFAULT));
	this->AllocateGPUMemory();
}

GRNG::~GRNG(){
	this->FreeGPUMemory();
	CheckStatus(curandDestroyGenerator(*this->generator));
}

float* GRNG::Get(unsigned count){
	size_t call_size = count * sizeof(float);
	if (call_size > this->alloc_size){
		printf("\n\tWarning:\ttoo large of call to RNG!\n");
		call_size = this->alloc_size;
	}
	CheckStatus(curandGenerateNormal(*this->generator, this->dev_r, call_size, this->mean, this->stddev));
	cudaDeviceSynchronize();
	return this->dev_r;
}

void GRNG::DisplayErrors(){
	if (this->status.size() > 0 || this->errorState.size() > 0){
		printf("\nGRNG ERRORS:\n");
		for (int i = 0; i < this->status.size(); i++){
			curandStatus_t stat = this->status[i];
			if (stat == CURAND_STATUS_ALLOCATION_FAILED)
				printf("Allocation of random numbers failed\n");
			else if (stat == CURAND_STATUS_ARCH_MISMATCH)
				printf("Architecture mismatch for random numbers\n");
			else if (stat == CURAND_STATUS_DOUBLE_PRECISION_REQUIRED)
				printf("Double precision requied for random numbers\n");
			else if (stat == CURAND_STATUS_INITIALIZATION_FAILED)
				printf("Initialization of random numbers failed\n");
			else if (stat == CURAND_STATUS_INTERNAL_ERROR)
				printf("Internal error generating random numbers\n");
			else if (stat == CURAND_STATUS_LAUNCH_FAILURE)
				printf("Launch failure generating random numbers\n");
			else if (stat == CURAND_STATUS_LENGTH_NOT_MULTIPLE)
				printf("Incorrect length requested for random numbers\n");
			else if (stat == CURAND_STATUS_NOT_INITIALIZED)
				printf("Container not initialized before generation\n");
			else if (stat == CURAND_STATUS_OUT_OF_RANGE)
				printf("Out of range error generating random numbers\n");
			else if (stat == CURAND_STATUS_PREEXISTING_FAILURE)
				printf("Preexisting error caused failure generating random numbers\n");
			else if (stat == CURAND_STATUS_TYPE_ERROR)
				printf("Type error generating random numbers\n");
			else if (stat == CURAND_STATUS_VERSION_MISMATCH)
				printf("Version mismatch encountered generating random numbers\n");
		}
		for (int i = 0; i < this->errorState.size(); i++)
			printf("%i -- %s\n", i, cudaGetErrorString(this->errorState[i]));

		this->errorState.empty();
		this->status.empty();
	}
}
// -------------------------------------------------------------------------
//	PRIVATE METHODS
// -------------------------------------------------------------------------
void GRNG::AllocateGPUMemory(){
	CheckSuccess(cudaMalloc((void**)&this->dev_r, this->alloc_size));
}

void GRNG::FreeGPUMemory(){
	CheckSuccess(cudaFree(this->dev_r));
}

void GRNG::CheckStatus(curandStatus_t stat){
	if (stat != CURAND_STATUS_SUCCESS) this->status.push_back(stat);
}

void GRNG::CheckSuccess(cudaError_t err){
	if (err != cudaSuccess) this->errorState.push_back(err);
}

#endif