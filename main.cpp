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

// Our application import
#include "particleSystem.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       64
#define NUM_PARTICLES   16384 * 32


extern "C" void cudaInit(int argc, char **argv);


int main(int argc, char *argv[])
{
	// Initialize CUDA
	cudaInit(argc, argv);

	// Simulation parameters
	int numParticles = NUM_PARTICLES;
	float timestep = 0.001f;
	float damping = 1.0f;
	float gravity = 0.0003f;
	int iterations = 1000;
	int ballr = 1;
	float collideSpring = 0.5f;;
	float collideDamping = 0.02f;;
	float collideShear = 0.1f;
	float collideAttraction = 0.0f;
	uint3 gridSize = make_uint3(GRID_SIZE, GRID_SIZE, GRID_SIZE);

	// Instantiate the system of particles
	auto psystem = new ParticleSystem(numParticles, gridSize);
	psystem->reset(ParticleSystem::CONFIG_RANDOM);

	// Pre time loop actions
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	cudaDeviceSynchronize();
    sdkStartTimer(&timer);

	// Run iterations
	for (int i = 0; i < iterations; ++i)
    {
        psystem->update(timestep);
    }

	// Post time loop actions
	cudaDeviceSynchronize();
    sdkStopTimer(&timer);

	// Print some speed metrics
	float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer)/(float)iterations);
	printf("particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-3 * numParticles)/fAvgSeconds, fAvgSeconds, numParticles, 1, 0);

	delete psystem;
	return 0;
}