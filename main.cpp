// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <string>

// Our application import
#include "particleSystem.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       64
#define NUM_PARTICLES   16384 * 4


extern "C" void cudaInit(int argc, char **argv);


void printInfo(StopWatchInterface *timer, int numParticles, int iterations, int iter)
{
	// Print some speed metrics
	float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer)/(float)iterations);
	printf("[%i / %i] particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n",
           iter, iterations, (1.0e-3 * numParticles)/fAvgSeconds, fAvgSeconds, numParticles, 1, 0);
}


int main(int argc, char *argv[])
{
	// Initialize CUDA
	cudaInit(argc, argv);

	// Simulation parameters
	const std::string outFile = "nw.xyz";
	int numParticles 		  = NUM_PARTICLES;
	uint filamentSize 		  = 32;
	float timestep 			  = 0.001f;
	int printRate 			  = 1000;
	int iterations 			  = 500000;
	float damping 			  = 0.0f;
	float gravity 			  = -0.001f;

	uint3 gridSize = make_uint3(GRID_SIZE, GRID_SIZE, GRID_SIZE);

	// Instantiate the system of particles
	auto psystem = new ParticleSystem(numParticles, gridSize);
	psystem->setGravity(gravity);
	psystem->setDamping(damping);
	psystem->setFilamentSize(filamentSize);
	psystem->reset(ParticleSystem::CONFIG_GRID);

	// Pre time loop actions
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	cudaDeviceSynchronize();
    sdkStartTimer(&timer);

	// Run iterations
	for (int i = 0; i < iterations; ++i)
    {
        psystem->update(timestep);

		if (i % printRate == 0)
		{
			psystem->saveToFile(outFile);
			printInfo(timer, numParticles, iterations, i);
		}
    }

	// Post time loop actions
	cudaDeviceSynchronize();
    sdkStopTimer(&timer);

	delete psystem;
	return 0;
}