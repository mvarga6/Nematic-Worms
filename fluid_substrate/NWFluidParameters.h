
#ifndef __FLUID_PARAMETERS_H__
#define __FLUID_PARAMETERS_H__

#include "cuda.h"
#include "cuda_runtime.h"
/* ------------------------------------------------------------------------
*	Data structure containing the parameters of a Fluid object.  Intended
*	to exist at same scope level and Fluid object because Fluid takes a
*	reference at initialization.
--------------------------------------------------------------------------*/
typedef struct {
	
	//.. default setup config
	int	_XDIM, _YDIM;

	//.. # of particles
	int	_NFLUID;

	//.. time steps between resetting neighbors list, max # of neighbors 
	int	_LISTSETGAP, _NMAX;

	//.. LJ energy and length scales
	float	_EPSILON, _SIGMA;

	//.. Langevin thermostat specs
	float	_KBT, _GAMMA;

	//.. LJ pre-calculations
	float	_SIGMA6, _2SIGMA6;
	float	_LJ_AMP;
	float	_RMIN, _R2MIN;
	float	_RCUT, _R2CUT;

	//.. buffer length setting neighbors lists
	float _BUFFER;

} FluidParameters;
/* ------------------------------------------------------------------------
*	This is the actual symbol that the parameters will be saved to
*	on the GPU device.  Accompanying this global parameter is the
*	function to allocate the parameters to this symbol.
--------------------------------------------------------------------------*/
__constant__ FluidParameters dev_flParams;
/* -----------------------------------------------------------------------
*	Function to allocate FluidParameters on GPU device.  Returns the
*	errors directly.  Basically acts as a typedef for cudaMemcpy(..)
--------------------------------------------------------------------------*/
cudaError_t ParametersToDevice(FluidParameters &params){
	return cudaMemcpyToSymbol(dev_flParams, &params, sizeof(FluidParameters));
}

#endif