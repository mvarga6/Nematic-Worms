
#ifndef __SIMULATION_PARAMETERS_H__
#define __SIMULATION_PARAMETERS_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include <string>
#include <cstdlib>
/* ------------------------------------------------------------------------
*	Data structure containing the parameters needed to run a simulation.  
*	Intended to exist inside a simulaltion object, NOT taken by reference
*	at initialization such as WormsParameters and FluidParameters.
----------------------------------------------------------------------------*/
typedef struct {

	//.. # of time steps once cross-linked
	int _NSTEPS;

	//.. number of inner time steps for fast potential calculations
	int _NSTEPS_INNER;

	//.. number of steps of equilization before cross-linking
	int _NSTEPS_EQUIL;

	//.. number of steps allowed for cross-linking
	int _NSTEPS_XLINK;

	//.. rate of printing frames, and max frames per output file
	int _FRAMERATE, _FRAMESPERFILE;

	//.. long time integration constant
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
/*--------------------------------------------------------------------------
*	Default values for all parameter values in SimulationParameters.
--------------------------------------------------------------------------*/
namespace DEFAULT {
	namespace SIM {
		static const int NSTEPS = 1000000;
		static const int NSTEPS_INNER = 10;
		static const int NSTEPS_EQUIL = 100000;
		static const int NSTEPS_XLINK = 50000;
		static const int FRAMERATE = 5000;
		static const int FRAMESPERFILE = 40;
		static const float DT = 0.01f;
		static const float XBOX = 2000.0f;
		static const float YBOX = 2000.0f;
		static const std::string FILENAME = "xlinker_simulation.xyz";
	}
}
//--------------------------------------------------------------------------
void GrabParameters(SimulationParameters * parameters, int argc, char *argv[], std::string &outfile){

	//.. cycle through arguments
	for (int i = 1; i < argc; i++){
		std::string arg = argv[i];
		std::string val;
		if (arg == "-dt"){
			if (i + 1 < argc){
				parameters->_DT = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-xbox"){
			if (i + 1 < argc){
				parameters->_XBOX = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-ybox"){
			if (i + 1 < argc){
				parameters->_YBOX = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-nsteps"){
			if (i + 1 < argc){
				parameters->_NSTEPS = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-nsteps-inner"){
			if (i + 1 < argc){
				parameters->_NSTEPS_INNER = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-nsteps-equil"){
			if (i + 1 < argc){
				parameters->_NSTEPS_EQUIL = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-nsteps-xlink"){
			if (i + 1 < argc){
				parameters->_NSTEPS_XLINK = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-framerate"){
			if (i + 1 < argc){
				parameters->_FRAMERATE = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-framesperfile"){
			if (i + 1 < argc){
				parameters->_FRAMESPERFILE = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-o"){
			if (i + 1 < argc){
				outfile = std::string(argv[1 + i++]);
			}
		}
	}
}
//--------------------------------------------------------------------------
void Init(SimulationParameters * parameters, int argc, char *argv[], std::string &outfile){

	//.. init with default parameters
	parameters->_DT = DEFAULT::SIM::DT;
	parameters->_XBOX = DEFAULT::SIM::XBOX;
	parameters->_YBOX = DEFAULT::SIM::YBOX;
	parameters->_NSTEPS = DEFAULT::SIM::NSTEPS;
	parameters->_NSTEPS_INNER = DEFAULT::SIM::NSTEPS_INNER;
	parameters->_NSTEPS_EQUIL = DEFAULT::SIM::NSTEPS_EQUIL;
	parameters->_FRAMERATE = DEFAULT::SIM::FRAMERATE;
	parameters->_FRAMESPERFILE = DEFAULT::SIM::FRAMESPERFILE;
	outfile = DEFAULT::SIM::FILENAME;

	//.. get assign cmdline parameters
	GrabParameters(parameters, argc, argv, outfile);

	//.. put on GPU and check for error
	cudaError_t err;
	err = ParametersToDevice(*parameters);
	std::cout << "\nSimulation parameters cudaMemcpyToSymbol returned:     \t" << cudaGetErrorString(err);
}
//--------------------------------------------------------------------------
#endif