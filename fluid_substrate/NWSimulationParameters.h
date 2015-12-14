
#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include <string>
/* ------------------------------------------------------------------------
*	Data structure containing the parameters needed to run a simulation.  
*	Intended to exist inside a simulaltion object, NOT taken by reference
*	at initialization such as WormsParameters and FluidParameters.
----------------------------------------------------------------------------*/
typedef struct {

	//..  name of simulation to be using in file i/o
	//std::string key;

	//.. # of time steps
	int _NSTEPS;

	//.. rate of printing frames, and max frames per output file
	int _FRAMERATE, _FRAMESPERFILE;

	//.. time integration constant
	float _DT;

	//..system physical size
	float _XBOX, _YBOX;

} SimulationParameters;
/* ------------------------------------------------------------------------
*	This is the actual symbol that the parameters will be saved to
*	on the GPU device.  Accompanying this global parameter is the
*	function to allocate the parameters to this symbol.
---------------------------------------------------------------------------*/
__constant__ SimulationParameters dev_simParams;
/* -------------------------------------------------------------------------
*	Function to allocate SimulationParameters on GPU device.  Returns 
*	the errors directly.  Basically acts as a typedef for cudaMemcpy(..)
----------------------------------------------------------------------------*/
cudaError_t ParametersToDevice(SimulationParameters &params){
	return cudaMemcpyToSymbol(dev_simParams, &params, sizeof(SimulationParameters));
}

#endif