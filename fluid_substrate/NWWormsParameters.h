
#ifndef __WORMS_PARAMETERS_H__
#define __WORMS_PARAMETERS_H__

#include "cuda.h"
#include "cuda_runtime.h"

/* ------------------------------------------------------------------------
*	Data structure containing the parameters of a Worms object.  Intended
*	to exist at same scope level and Worms object because Worms takes a
*	reference at initialization.
--------------------------------------------------------------------------*/
typedef struct {

	//.. default setup config
	int _XDIM, _YDIM;

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

/*
*	Namespace holding all available parameters for class Worms 
*	stored in constant memory on the gpu device.
*/
/*namespace wp{
	__constant__ int _XDIM;
	__constant__ int _YDIM;
	__constant__ int _NP;
	__constant__ int _NWORMS;
	__constant__ int _NPARTICLES;
	__constant__ int _NMAX;
	__constant__ float _EPSILON;
	__constant__ float _SIGMA;
	__constant__ float _DRIVE;
	__constant__ float _K1;
	__constant__ float _K2;
	__constant__ float _K3;
	__constant__ float _L1;
	__constant__ float _L2;
	__constant__ float _L3;
	__constant__ float _KBT;
	__constant__ float _GAMMA;
	__constant__ float _DAMP;
	__constant__ float _SIGMA6;
	__constant__ float _2SIGMA6;
	__constant__ float _LJ_AMP;
	__constant__ float _RMIN;
	__constant__ float _R2MIN;
	__constant__ float _RCUT;
	__constant__ float _R2CUT;
	__constant__ float _BUFFER;
}

class WormsParameters {

	//.. contains errors
	std::vector<cudaError_t> errorState;

	//.. all the parameters
	int cm_XDIM;
	int cm_YDIM;
	int cm_NP;
	int cm_NWORMS;	
	int cm_NPARTICLES;
	int cm_LISTSETGAP;	
	int cm_NMAX;
	float cm_EPSILON;
	float cm_SIGMA;
	float cm_DRIVE;
	float cm_K1;
	float cm_K2;
	float cm_K3;
	float cm_L1;
	float cm_L2;
	float cm_L3;
	float cm_KBT;
	float cm_GAMMA;
	float cm_DAMP;
	float cm_SIGMA6;
	float cm_2SIGMA6;
	float cm_LJ_AMP;
	float cm_RMIN;
	float cm_R2MIN;
	float cm_RCUT;
	float cm_R2CUT;
	float cm_BUFFER;

	//.. for access from worms
	friend class Worms;

public:
	WormsParameters();
	~WormsParameters();

	bool LoadFromFile(std::string &filename);
	void SaveToFile(std::string &filename);
	void SymbolsToDevice();
	
	int nparticles(){ return cm_NPARTICLES; }
	int nworms(){ return cm_NWORMS; }
	int np(){ return cm_NP; }
	float l1(){ return cm_L1; }
	float rmin(){ return cm_RMIN; }

	void SetParamGroupA(
		const int &numPerWorm,
		const int &numWorms,
		const int &initXdim,
		const int &initYdim);

	void SetParamGroupB(
		const float &K1,
		const float &K2,
		const float &K3,
		const float &L1,
		const float &L2,
		const float &L3);

	void SetParamGroupC(
		const float &kbt,
		const float &epsilon,
		const float &sigma,
		const float &gamma,
		const float &damping,
		const float &rcut);

	void SetParamGroupD(
		const int &listSetGap,
		const int &maxNeighbors,
		const float &neighborSearchBuffer);

private:

	void checkSuccess(cudaError_t err);
	void calculateSymbols();

};
// -------------------------------------------------------------------------
//	PUBLIC METHODS
// -------------------------------------------------------------------------
WormsParameters::WormsParameters(){

}

WormsParameters::~WormsParameters(){

}

bool WormsParameters::LoadFromFile(std::string &filename){
	return true;
}

void WormsParameters::SaveToFile(std::string &filename){

}

void WormsParameters::SymbolsToDevice(){

	//.. recalculate symbol values
	this->calculateSymbols();

	//.. integer symbols
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_NP, &cm_NP, sizeof(int)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_NWORMS, &cm_NWORMS, sizeof(int)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_XDIM, &cm_XDIM, sizeof(int)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_YDIM, &cm_YDIM, sizeof(int)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_NPARTICLES, &cm_NPARTICLES, sizeof(int)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_NMAX, &cm_NMAX, sizeof(int)));

	//.. float symbols
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_EPSILON, &cm_EPSILON, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_SIGMA, &cm_SIGMA, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_DRIVE, &cm_DRIVE, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_K1, &cm_K1, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_K2, &cm_K2, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_K3, &cm_K3, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_L1, &cm_L2, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_L2, &cm_L2, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_L3, &cm_L3, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_KBT, &cm_KBT, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_GAMMA, &cm_GAMMA, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_DAMP, &cm_DAMP, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_SIGMA6, &cm_SIGMA6, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_2SIGMA6, &cm_2SIGMA6, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_LJ_AMP, &cm_LJ_AMP, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_RMIN, &cm_RMIN, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_R2MIN, &cm_R2MIN, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_RCUT, &cm_RCUT, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_R2CUT, &cm_R2CUT, sizeof(float)));
	checkSuccess(cudaMemcpyToSymbol((void*)&wp::_BUFFER, &cm_BUFFER, sizeof(float)));
}

void WormsParameters::SetParamGroupA(
	const int &numPerWorm,
	const int &numWorms,
	const int &initXdim,
	const int &initYdim){
	this->cm_NP = numPerWorm;
	this->cm_NWORMS = numWorms;
	this->cm_XDIM = initXdim;
	this->cm_YDIM = initYdim;
	this->calculateSymbols();
}

void WormsParameters::SetParamGroupB(
	const float &K1,
	const float &K2,
	const float &K3,
	const float &L1,
	const float &L2,
	const float &L3){
	this->cm_K1 = K1;
	this->cm_K2 = K2;
	this->cm_K3 = K3;
	this->cm_L1 = L1;
	this->cm_L2 = L2;
	this->cm_L3 = L3;
}

void WormsParameters::SetParamGroupC(
	const float &kbt,
	const float &epsilon,
	const float &sigma,
	const float &gamma,
	const float &dampingCoef,
	const float &rcut){
	this->cm_KBT = kbt;
	this->cm_EPSILON = epsilon;
	this->cm_SIGMA = sigma;
	this->cm_GAMMA = gamma;
	this->cm_DAMP = dampingCoef;
	this->cm_RCUT = rcut;
	this->calculateSymbols();
}

void WormsParameters::SetParamGroupD(
	const int &listSetGap,
	const int &maxNeighbors,
	const float &neighborSearchBuffer){
	this->cm_LISTSETGAP = listSetGap;
	this->cm_NMAX = maxNeighbors;
	this->cm_BUFFER = neighborSearchBuffer;
}

	// -------------------------------------------------------------------------
	//	PRIVATE METHODS
	// -------------------------------------------------------------------------

void WormsParameters::checkSuccess(cudaError_t err){
	if (err != cudaSuccess) this->errorState.push_back(err);
}

void WormsParameters::calculateSymbols(){
	this->cm_NPARTICLES = this->cm_NP * this->cm_NWORMS;
	this->cm_SIGMA6 = powf(this->cm_SIGMA, 6);
	this->cm_2SIGMA6 = 2.0f * this->cm_SIGMA6;
	this->cm_LJ_AMP = 24.0f * this->cm_EPSILON * this->cm_SIGMA6;
	this->cm_RMIN = _216 * this->cm_SIGMA;
	this->cm_R2MIN = this->cm_RMIN * this->cm_RMIN;
	this->cm_R2CUT = this->cm_RCUT * this->cm_RCUT;
}*/

#endif