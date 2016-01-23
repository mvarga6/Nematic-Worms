
#ifndef __WORMS_PARAMETERS_H__
#define __WORMS_PARAMETERS_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "NWConstants.h"
#include <string>
#include <cstdlib>
/* ------------------------------------------------------------------------
*	Data structure containing the parameters of a Worms object.  Intended
*	to exist at same scope level and Worms object because Worms takes a
*	reference at initialization.
--------------------------------------------------------------------------*/
typedef struct {

	//.. default setup config
	int _XDIM, _YDIM, _ZDIM;

	//.. particles per worm, # of worms, total particle #
	int _NP, _NWORMS, _NPARTICLES;

	//.. scales the landscape forces
	float _LANDSCALE;

	//.. time steps between reseting neighbors lists, maximum # of neighbors
	int _LISTSETGAP, _NMAX;

	//.. LJ energy and length scale, activity scalor
	float _EPSILON, _SIGMA, _DRIVE;

	//.. spring constants in worms
	float _K1, _K2, _K3, _Ka;

	//.. spring equilibrium lengths
	float _L1, _L2, _L3;

	//.. Langevin thermostat specs
	float _KBT, _GAMMA, _DAMP;

	//.. pre-calculations for LJ
	float _SIGMA6, _2SIGMA6, _LJ_AMP;
	float _RMIN, _R2MIN;
	float _RCUT, _R2CUT;

	//.. buffer length when setting neighbors lists
	float _BUFFER;

} WormsParameters;
/* ------------------------------------------------------------------------
*	This is the actual symbol that the parameters will be saved to 
*	on the GPU device.  Accompanying this global parameter is the 
*	function to allocate the parameters to this symbol.
--------------------------------------------------------------------------*/
__constant__ WormsParameters dev_Params;
/* ------------------------------------------------------------------------
*	Function to allocate WormsParameters on GPU device.  Returns the
*	errors directly.  Basically acts as a typedef for cudaMemcpy(..)
--------------------------------------------------------------------------*/
cudaError_t ParametersToDevice(WormsParameters &params){
	return cudaMemcpyToSymbol(dev_Params, &params, sizeof(WormsParameters));
}
/*------------------------------------------------------------------------
*	Default values for all parameter values in WormsParameters.
--------------------------------------------------------------------------*/
namespace DEFAULT {
	namespace WORMS {
		static const int	XDIM = 5;
		static const int	YDIM = 40;
		static const int	ZDIM = 2;
		static const int	NP = 10;
		static const int	NWORMS = XDIM * YDIM * ZDIM;
		static const int	NPARTICLES = NP * NWORMS;
		static const int	LISTSETGAP = 50;
		static const int	NMAX = 128;
		static const float	EPSILON = 0.2f;
		static const float	SIGMA = 1.0f;
		static const float	DRIVE = 1.0f;
		static const float	K1 = 57.146f;
		static const float	K2 = 10.0f * K1;
		static const float	K3 = 2.0f * K2 / 3.0f;
		static const float	Ka = 5.0f;
		static const float	L1 = 0.8f;
		static const float	L2 = 1.6f;
		static const float	L3 = 2.4f;
		static const float	KBT = 0.05f;
		static const float	GAMMA = 2.0f;
		static const float	DAMP = 3.0f;
		static const float	BUFFER = 0.5f;
		static const float	LANDSCALE = 1.0f;
	}
}
//----------------------------------------------------------------------------
void CalculateParameters(WormsParameters * parameters, bool WCA = false){
	parameters->_NWORMS = parameters->_XDIM * parameters->_YDIM * parameters->_ZDIM;
	parameters->_NPARTICLES = parameters->_NP * parameters->_NWORMS;
	parameters->_SIGMA6 = powf(parameters->_SIGMA, 6.0f);
	parameters->_2SIGMA6 = 2.0f * parameters->_SIGMA6;
	parameters->_LJ_AMP = 24.0f * parameters->_EPSILON * parameters->_SIGMA6;
	parameters->_RMIN = nw::_216 * parameters->_SIGMA;
	parameters->_R2MIN = parameters->_RMIN * parameters->_RMIN;
	if (WCA)
		parameters->_RCUT = parameters->_RMIN;
	else
		parameters->_RCUT = 2.5f * parameters->_SIGMA;
	parameters->_R2CUT = parameters->_RCUT * parameters->_RCUT;
}
//----------------------------------------------------------------------------
void GrabParameters(WormsParameters * parameters, int argc, char *argv[], bool &wca){

	//.. cycle through arguments
	for (int i = 1; i < argc; i++){
		std::string arg = argv[i];
		std::string val;
		if (arg == "-xdim"){
			if (i + 1 < argc){
				//std::string val = argv[1 + i++];
				parameters->_XDIM = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-ydim"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_YDIM = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-zdim"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_ZDIM = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-np"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_NP = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-listsetgap"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_LISTSETGAP = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-nmax"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_NMAX = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-epsilon"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_EPSILON = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-sigma"){
			if (i + 1 < argc){
				///val = std::string(argv[1 + i++]);
				parameters->_SIGMA = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-drive"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_DRIVE = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-k1"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_K1 = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-k2"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_K2 = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-k3"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_EPSILON = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-ka"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_Ka = std::strtof(argv[1 + i++], NULL);
				printf("\nKa changed: %f", parameters->_Ka);
			}
		}
		else if (arg == "-l1"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_L1 = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-l2"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_L2 = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-l3"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_L3 = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-kbt"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_KBT = std::strtof(argv[1 + i++], NULL);
				printf("\nKBT changed: %f", parameters->_KBT);
			}
		}
		else if (arg == "-gamma"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_GAMMA = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-damp"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_DAMP = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-buffer"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_BUFFER = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-landscale"){
			if (i + 1 < argc){
				//val = std::string(argv[1 + i++]);
				parameters->_LANDSCALE = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-wca"){
			wca = true;
		}
	}
}
//--------------------------------------------------------------------------
//.. initialization function
void Init(WormsParameters * parameters, int argc, char *argv[], bool WCA = false){
	parameters->_XDIM = DEFAULT::WORMS::XDIM;
	parameters->_YDIM = DEFAULT::WORMS::YDIM;
	parameters->_ZDIM = DEFAULT::WORMS::ZDIM;
	parameters->_NP = DEFAULT::WORMS::NP;
	//parameters->_NWORMS = DEFAULT::WORMS::NWORMS;
	//parameters->_NPARTICLES = DEFAULT::WORMS::NPARTICLES;
	parameters->_LISTSETGAP = DEFAULT::WORMS::LISTSETGAP;
	parameters->_NMAX = DEFAULT::WORMS::NMAX;
	parameters->_EPSILON = DEFAULT::WORMS::EPSILON;
	parameters->_SIGMA = DEFAULT::WORMS::SIGMA;
	parameters->_DRIVE = DEFAULT::WORMS::DRIVE;
	parameters->_K1 = DEFAULT::WORMS::K1;
	parameters->_K2 = DEFAULT::WORMS::K2;
	parameters->_K3 = DEFAULT::WORMS::K3;
	parameters->_Ka = DEFAULT::WORMS::Ka;
	parameters->_L1 = DEFAULT::WORMS::L1;
	parameters->_L2 = DEFAULT::WORMS::L2;
	parameters->_L3 = DEFAULT::WORMS::L3;
	parameters->_KBT = DEFAULT::WORMS::KBT;
	parameters->_GAMMA = DEFAULT::WORMS::GAMMA;
	parameters->_DAMP = DEFAULT::WORMS::DAMP;
	parameters->_BUFFER = DEFAULT::WORMS::BUFFER;
	parameters->_LANDSCALE = DEFAULT::WORMS::LANDSCALE;
	
	GrabParameters(parameters, argc, argv, WCA);
	CalculateParameters(parameters, WCA);

	cudaError_t err;
	err = ParametersToDevice(*parameters);
	std::cout << "\nWorms parameters cudaMemcpyToSymbol returned:\t" << cudaGetErrorString(err);
}
//--------------------------------------------------------------------------
#endif