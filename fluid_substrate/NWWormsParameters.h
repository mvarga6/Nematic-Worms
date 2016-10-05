
#ifndef __WORMS_PARAMETERS_H__
#define __WORMS_PARAMETERS_H__
// 2D
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

	//.. steric interaction on/off
	bool _NOINT;

	//.. drive type
	bool _EXTENSILE;

	//.. spring constants in worms
	float _K1, _Ka;

	//.. spring equilibrium lengths
	float _L1;

	//.. Langevin thermostat specs
	float _KBT, _GAMMA, _DAMP;

	//.. pre-calculations for LJ
	float _SIGMA6, _2SIGMA6, _LJ_AMP;
	float _RMIN, _R2MIN;
	float _RCUT, _R2CUT;

	//.. buffer length when setting neighbors lists
	float _BUFFER;

	//.. cell size for neighbor finding
	float _DCELL;

	//.. flag for random adhering distribute
	bool _RAD;

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
cudaError_t ParametersToDevice(WormsParameters &params, bool attractivePotentialCutoff = false){

	//.. pre-calculated variables
	//params._SIGMA6 = powf(params._SIGMA, 6.0f);
	//params._2SIGMA6 = 2.0f * params._SIGMA6;
	//params._RMIN = powf(2.0f, 1.0f / 6.0f) * params._SIGMA;
	//params._R2MIN = params._RMIN * params._RMIN;
	//if (attractivePotentialCutoff) params._RCUT = 2.5f * params._SIGMA;
	//else params._RCUT = params._RMIN;
	//params._R2MIN = params._RMIN * params._RMIN;
	//params._LJ_AMP = 24.0f * params._EPSILON * params._SIGMA6;
	return cudaMemcpyToSymbol(dev_Params, &params, sizeof(WormsParameters));
}
/*------------------------------------------------------------------------
*	Default values for all parameter values in WormsParameters.
--------------------------------------------------------------------------*/
namespace DEFAULT {
	namespace WORMS {
		static const int	XDIM = 5;
		static const int	YDIM = 20;
		static const int	ZDIM = 1;
		static const int	NP = 10;
		static const int	NWORMS = XDIM * YDIM * ZDIM;
		static const int	NPARTICLES = NP * NWORMS;
		static const int	LISTSETGAP = 25;
		static const int	NMAX = 96;
		static const float	EPSILON = 0.5f;
		static const float	SIGMA = 1.0f;
		static const float	DRIVE = 1.0f;
		static const float	DRIVE_ROT = 0.0f;
		static const float	K1 = 57.146f;
		static const float	Ka = 5.0f;
		static const float	L1 = 0.8f;
		static const float	KBT = 1.0f;
		static const float	GAMMA = 2.0f;
		static const float	DAMP = 3.0f;
		static const float	BUFFER = 3.0f;
		static const float	LANDSCALE = 0.0f;
		static const float	Kx = 10.0f;
		static const float	DCELL = 3.0f;
		static const bool	RAD = false;
		static const bool   EXTENSILE = false;
		static const bool	NOINT = false;
	}
}
//----------------------------------------------------------------------------
void CalculateParameters(WormsParameters * parameters, bool WCA = false){
	if (_D_ == 2) parameters->_ZDIM = 1; // ensure one layer if 2d
	parameters->_NWORMS = parameters->_XDIM * parameters->_YDIM * parameters->_ZDIM; // calculate # of worms
	parameters->_NPARTICLES = parameters->_NP * parameters->_NWORMS; // calculate # of particles
	parameters->_SIGMA6 = powf(parameters->_SIGMA, 6.0f); // LJ scalor
	parameters->_2SIGMA6 = 2.0f * parameters->_SIGMA6; // LJ scalor
	parameters->_LJ_AMP = 24.0f * parameters->_EPSILON * parameters->_SIGMA6; // LJ scalor
	parameters->_RMIN = nw::constants::_216 * parameters->_SIGMA; // calculate LJ minimum
	parameters->_R2MIN = parameters->_RMIN * parameters->_RMIN; // calculate sqr minimum
	if (WCA) // set Weeks-Chandeler-Anderson cutoff
		parameters->_RCUT = parameters->_RMIN;
	else // set full Lennard-Jones cutoff
		parameters->_RCUT = 2.5f * parameters->_SIGMA;
	parameters->_DCELL = parameters->_RCUT + parameters->_BUFFER; // linked-list cell width
	parameters->_R2CUT = parameters->_RCUT * parameters->_RCUT; // LJ cutoff distance
}
//----------------------------------------------------------------------------
void GrabParameters(WormsParameters * parameters, int argc, char *argv[], bool &wca, bool &xramp){

	//.. cycle through arguments
	for (int i = 1; i < argc; i++){
		std::string arg = argv[i];
		std::string val;
		if (arg == "-xdim"){
			if (i + 1 < argc){
				parameters->_XDIM = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-ydim"){
			if (i + 1 < argc){
				parameters->_YDIM = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-zdim"){
			if (i + 1 < argc){
				parameters->_ZDIM = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-np"){
			if (i + 1 < argc){
				parameters->_NP = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-listsetgap"){
			if (i + 1 < argc){
				parameters->_LISTSETGAP = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-nmax"){
			if (i + 1 < argc){
				parameters->_NMAX = (int)std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-epsilon"){
			if (i + 1 < argc){
				parameters->_EPSILON = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-sigma"){
			if (i + 1 < argc){
				parameters->_SIGMA = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-drive"){
			if (i + 1 < argc){
				parameters->_DRIVE = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-k1"){
			if (i + 1 < argc){
				parameters->_K1 = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-ka"){
			if (i + 1 < argc){
				parameters->_Ka = std::strtof(argv[1 + i++], NULL);
				printf("\nKa changed: %f", parameters->_Ka);
			}
		}
		else if (arg == "-l1"){
			if (i + 1 < argc){
				parameters->_L1 = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-kbt"){
			if (i + 1 < argc){
				parameters->_KBT = std::strtof(argv[1 + i++], NULL);
				printf("\nKBT changed: %f", parameters->_KBT);
			}
		}
		else if (arg == "-gamma"){
			if (i + 1 < argc){
				parameters->_GAMMA = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-damp"){
			if (i + 1 < argc){
				parameters->_DAMP = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-buffer"){
			if (i + 1 < argc){
				parameters->_BUFFER = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-A" || arg == "-landscale"){
			if (i + 1 < argc){
				parameters->_LANDSCALE = std::strtof(argv[1 + i++], NULL);
			}
		}
		else if (arg == "-wca"){
			wca = true;
			printf("\nUsing Weeks-Chandler-Anderson potential");
		}
		else if (arg == "-rad"){
			parameters->_RAD = true;
			printf("\nInitializing worms using Random Adhersion");
		}
		else if (arg == "-extensile"){
			parameters->_EXTENSILE = true;
			printf("\nUsing extensile driving mechanism");
		}
		else if (arg == "-noint"){
			parameters->_NOINT = true;
			printf("\nExcluding steric interaction between worms");
		}
	}
}
//--------------------------------------------------------------------------
//.. initialization function
void Init(WormsParameters * parameters, int argc, char *argv[]){
	bool WCA = false, XRAMP = false; // flags
	parameters->_XDIM = DEFAULT::WORMS::XDIM;
	parameters->_YDIM = DEFAULT::WORMS::YDIM;
	parameters->_ZDIM = DEFAULT::WORMS::ZDIM;
	parameters->_NP = DEFAULT::WORMS::NP;
	parameters->_LISTSETGAP = DEFAULT::WORMS::LISTSETGAP;
	parameters->_NMAX = DEFAULT::WORMS::NMAX;
	parameters->_EPSILON = DEFAULT::WORMS::EPSILON;
	parameters->_SIGMA = DEFAULT::WORMS::SIGMA;
	parameters->_NOINT = DEFAULT::WORMS::NOINT;
	parameters->_DRIVE = DEFAULT::WORMS::DRIVE;
	parameters->_EXTENSILE = DEFAULT::WORMS::EXTENSILE;
	parameters->_K1 = DEFAULT::WORMS::K1;
	parameters->_Ka = DEFAULT::WORMS::Ka;
	parameters->_L1 = DEFAULT::WORMS::L1;
	parameters->_KBT = DEFAULT::WORMS::KBT;
	parameters->_GAMMA = DEFAULT::WORMS::GAMMA;
	parameters->_DAMP = DEFAULT::WORMS::DAMP;
	parameters->_BUFFER = DEFAULT::WORMS::BUFFER;
	parameters->_LANDSCALE = DEFAULT::WORMS::LANDSCALE;
	parameters->_DCELL = DEFAULT::WORMS::DCELL;
	parameters->_RAD = DEFAULT::WORMS::RAD;
	
	GrabParameters(parameters, argc, argv, WCA, XRAMP);
	CalculateParameters(parameters, WCA);

	cudaError_t err;
	err = ParametersToDevice(*parameters);
	std::cout << "\nWorms parameters cudaMemcpyToSymbol returned:\t" << cudaGetErrorString(err);
}
//--------------------------------------------------------------------------
#endif