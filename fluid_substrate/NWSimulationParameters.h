
#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

#include "cuda.h"
#include "cuda_runtime.h"
/* ------------------------------------------------------------------------
*	Data structure containing the parameters needed to run a simulation.  
*	Intended to exist inside a simulaltion object, NOT taken by reference
*	at initialization such as WormsParameters and FluidParameters.
----------------------------------------------------------------------------*/
typedef struct {

	//.. # of time steps
	int _NSTEPS;

	//.. rate of printing frames, and max frames per output file
	int _FRAMERATE, _FRAMESPERFILE;

	//.. long time integration constant
	float _DT;

	//.. number of inner time steps for fast potential calculations
	int _NSTEPS_INNER;

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
/*--------------------------------------------------------------------------
*	Default values for all parameter values in SimulationParameters.
--------------------------------------------------------------------------*/
namespace DEFAULT {
	namespace SIM {
		static const int NSTEPS = 100000;
		static const int NSTEPS_INNER = 10;
		static const int FRAMERATE = 1000;
		static const int FRAMESPERFILE = 100;
		static const float DT = 0.01f;
		static const float XBOX = 100.0f;
		static const float YBOX = 100.0f;
	}
}
//--------------------------------------------------------------------------
void Init(SimulationParameters * parameters){
	parameters->_DT = DEFAULT::SIM::DT;
	parameters->_XBOX = DEFAULT::SIM::XBOX;
	parameters->_YBOX = DEFAULT::SIM::YBOX;
	parameters->_NSTEPS = DEFAULT::SIM::NSTEPS;
	parameters->_NSTEPS_INNER = DEFAULT::SIM::NSTEPS_INNER;
	parameters->_FRAMERATE = DEFAULT::SIM::FRAMERATE;
	parameters->_FRAMESPERFILE = DEFAULT::SIM::FRAMESPERFILE;

	cudaError_t err;
	err = ParametersToDevice(*parameters);
	std::cout << "Simulation parameters cudaMemcpyToSymbol returned:     \t" << cudaGetErrorString(err) << std::endl;
}
//--------------------------------------------------------------------------
#endif