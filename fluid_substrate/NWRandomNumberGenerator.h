
#ifndef __RANDOM_NUMBER_GENERATOR_H__
#define __RANDOM_NUMBER_GENERATOR_H__

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
	//float * dev_rx;
	//float * dev_ry;

	//.. actual number generator
	curandGenerator_t * generator;

	//.. status of the generator
	std::vector<curandStatus_t> status;
	std::vector<cudaError_t> errorState;

	//.. size of allocation on device
	size_t alloc_size;

	//.. Gaussian distribution specs
	const float mean;
	const float stddev;

public:
	GRNG(int maxCallSize, float distributionMean, float standardDeviation);
	~GRNG();

	float* Get(unsigned count);
	void PointToRandom(float *ptrToRx, float *ptrToRy);
	void DisplayErrors();

private:
	void AllocateGPUMemory(int maxRequested);
	void FreeGPUMemory();
	void CheckStatus(curandStatus_t stat);
	void CheckSuccess(cudaError_t err);
};
// ---------------------------------------------------------------------------
//	PUBLIC METHODS
// ---------------------------------------------------------------------------
GRNG::GRNG(int maxCallSize, float distributionMean, float standardDeviation)
	: generator(new curandGenerator_t), dev_r(NULL), /*dev_rx(NULL), dev_ry(NULL),*/
	mean(distributionMean), stddev(standardDeviation){

	this->AllocateGPUMemory(maxCallSize);
	CheckStatus(curandCreateGenerator(this->generator, CURAND_RNG_PSEUDO_DEFAULT));
	CheckStatus(curandSetPseudoRandomGeneratorSeed(*this->generator, 2345));
	printf("Random number generater created with %i errors.\n", this->status.size() + this->errorState.size());
}

GRNG::~GRNG(){
	this->FreeGPUMemory();
	CheckStatus(curandDestroyGenerator(*this->generator));
}

float* GRNG::Get(unsigned count){
	size_t call_size = count * sizeof(float);
	if (call_size > this->alloc_size){
		printf("\n\tWarning:\ttoo large of call to RNG!\n");
		count = this->alloc_size / sizeof(float);
	}
	CheckStatus(curandGenerateNormal(*this->generator, this->dev_r, count, this->mean, this->stddev));
	cudaDeviceSynchronize();
	//cudaMemset(this->dev_r, 0, this->alloc_size);
	return this->dev_r;
}

void GRNG::PointToRandom(float *ptrToRx, float *ptrToRy){
	//CheckStatus(curandGenerateNormal(*(this->generator), this->dev_rx, ))
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
void GRNG::AllocateGPUMemory(int maxRequested){

	float ratio = float(maxRequested) / 32.0f;
	float incr = float(ceilf(ratio)) - ratio;
	this->alloc_size = sizeof(float) * (maxRequested + int(ceilf(incr*32.0f)));

	printf("Memory requested for RNG:\t\t%i\n", sizeof(float)*maxRequested);
	printf("Adjusted to increment of warp size:\t%i\n", this->alloc_size);

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