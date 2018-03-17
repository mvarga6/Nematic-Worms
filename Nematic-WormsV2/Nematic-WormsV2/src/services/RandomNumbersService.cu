#include "..\..\include\RandomNumbersService.h"


NW::RandomNumbersService::RandomNumbersService(int maxCallSize, real distributionMean, real standardDeviation)
	: generator(new curandGenerator_t), dev_n(NULL),
	alloc_size(sizeof(float)*maxCallSize), mean(distributionMean),
	stddev(standardDeviation), max_size(maxCallSize) {

	CheckStatus(curandCreateGenerator(this->generator, CURAND_RNG_PSEUDO_DEFAULT));

	this->Threads_Per_Block = 256;
	this->Blocks_Per_Kernel = (this->max_size / this->Threads_Per_Block) + 1;
	this->AllocateGPUMemory(this->max_size);
	//initKernel <<<this->Blocks_Per_Kernel, this->Threads_Per_Block >>> (this->dev_s, this->max_size);
	printf("\nRandom Number Generator (GRNG) Created");
}
//-------------------------------------------------------------------------------------------
NW::RandomNumbersService::~RandomNumbersService() {
	this->FreeGPUMemory();
	CheckStatus(curandDestroyGenerator(*this->generator));
}
//-------------------------------------------------------------------------------------------
real* NW::RandomNumbersService::Get(unsigned count, bool uniform = false) {
	size_t call_size = count * sizeof(float);
	if (call_size > this->alloc_size) {
		printf("\n\tWarning:\ttoo large of call to RNG!\n");
		count = this->alloc_size / sizeof(real);
	}
	if (uniform)
		CheckStatus(curandGenerateUniform(*this->generator, this->dev_n, count));
	else
		CheckStatus(curandGenerateNormal(*this->generator, this->dev_n, count, this->mean, this->stddev));
	CheckSuccess(cudaDeviceSynchronize());
	return this->dev_n;
}
//-------------------------------------------------------------------------------------------
void NW::RandomNumbersService::DisplayErrors() {
	if (this->status.size() > 0 || this->errorState.size() > 0) {
		printf("\nGRNG ERRORS:\n");
		for (int i = 0; i < this->status.size(); i++) {
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
void NW::RandomNumbersService::AllocateGPUMemory(int N) {
	CheckSuccess(cudaMalloc((void**)&this->dev_n, this->alloc_size));
	//CheckSuccess(cudaMalloc((void**)&this->dev_s, N*sizeof(curandState)));
}
//-------------------------------------------------------------------------------------------
void NW::RandomNumbersService::FreeGPUMemory() {
	CheckSuccess(cudaFree(this->dev_n));
	//CheckSuccess(cudaFree(this->dev_s));
}
//-------------------------------------------------------------------------------------------
void NW::RandomNumbersService::CheckStatus(curandStatus_t stat) {
	if (stat != CURAND_STATUS_SUCCESS) this->status.push_back(stat);
}
//-------------------------------------------------------------------------------------------
void NW::RandomNumbersService::CheckSuccess(cudaError_t err) {
	if (err != cudaSuccess) this->errorState.push_back(err);
}
