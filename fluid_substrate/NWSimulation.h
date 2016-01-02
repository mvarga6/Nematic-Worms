#ifndef __SIMULATION_H__
#define __SIMULATION_H__
// -----------------------------------------
////////	NEMATIC WORMS PROJECT	////////
// -----------------------------------------
#include "NWmain.h"
#include "NWWorms.h"
#include "NWRandomNumberGenerator.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"
#include "NWKernels.h"
// --------------------------------------------
////////	CLASS DEFINING A SIMULATION ///////
// --------------------------------------------
class NWSimulation {

	//.. simulation components
	Worms				 * worms;
	WormsParameters		 * params;
	GRNG				 * rng;
	SimulationParameters * simparams;

	//.. file stream
	std::ofstream fxyz;

	//.. time
	float time;

public:
	NWSimulation();
	~NWSimulation();
	void Run();

private:
	void XYZPrint(int itime);
	void DisplayErrors();

};
// ---------------------------------------------
////////	IMPLEMENTATION HERE			////////
// ---------------------------------------------

NWSimulation::NWSimulation(){

	//.. setup parameters (should be done with cmdline input)
	this->params = new WormsParameters();
	this->params->_XDIM = 10;
	this->params->_YDIM = 10;
	this->params->_NP = 10;
	this->params->_NWORMS = params->_XDIM * params->_YDIM;
	this->params->_NPARTICLES = params->_NP * params->_NWORMS;
	this->params->_LISTSETGAP = 200;
	this->params->_NMAX = 28;
	this->params->_EPSILON = 1.0f;
	this->params->_SIGMA = 1.0f;
	this->params->_DRIVE = 1.0f;
	this->params->_K1 = 57.146f;
	this->params->_K2 = 10.0f * params->_K1;
	this->params->_K3 = 2.0f * params->_K2 / 3.0f;
	this->params->_L1 = 0.8f;
	this->params->_L2 = 2.0f * params->_L1;
	this->params->_L3 = 3.0f * params->_L1;
	this->params->_KBT = 0.0001f;
	this->params->_GAMMA = 2.0f;
	this->params->_DAMP = 3.0f;
	this->params->_SIGMA6 = powf(params->_SIGMA, 6.0f);
	this->params->_2SIGMA6 = params->_SIGMA6 * 2.0f;
	this->params->_LJ_AMP = 24.0f * params->_EPSILON * params->_SIGMA6;
	this->params->_RMIN = _216 * params->_SIGMA;
	this->params->_R2MIN = params->_RMIN * params->_RMIN;
	this->params->_RCUT = params->_RMIN; // 2.5f * params->_SIGMA;
	this->params->_R2CUT = params->_RCUT * params->_RCUT;
	this->params->_BUFFER = 1.5f;
	this->params->_LANDSCALE = 10.0f;

	//.. setup simulation parameters
	this->simparams = new SimulationParameters();
	this->simparams->_DT = 0.001f;
	this->simparams->_FRAMERATE = 1000;
	this->simparams->_FRAMESPERFILE = 200;
	this->simparams->_NSTEPS = 1000000;
	this->simparams->_XBOX = 100.0f;
	this->simparams->_YBOX = 40.0f;
	this->time = 0.0f;

	//.. parameters to device
	cudaError_t err;
	err = ParametersToDevice(*params);
	std::cout << "Worms parameters cudaMemcpyToSymbol returned:     \t" << cudaGetErrorString(err) << std::endl;
	err = ParametersToDevice(*simparams);
	std::cout << "Simulation parameters cudaMemcpyToSymbol returned:\t" << cudaGetErrorString(err) << std::endl;

	CheckParametersOnDevice <<< 1, 1 >>>();
	ErrorHandler(cudaDeviceSynchronize());

	//.. setup random number generator
	this->rng = new GRNG(3 * params->_NPARTICLES, 0.0f, 1.0f);

	//.. define the worms
	this->worms = new Worms();
}

NWSimulation::~NWSimulation(){
	delete this->worms;
	delete this->rng;
	delete this->params;
	delete this->simparams;
}

void NWSimulation::Run(){
	this->worms->Init(this->rng, this->params, this->simparams);
	//this->DisplayErrors();
	this->fxyz.open("output//3d_test3.xyz");

	this->XYZPrint(0);
	this->worms->ResetNeighborsList(0);
	//this->worms->DislayThetaPhi();
	for (int itime = 0; itime < this->simparams->_NSTEPS; itime++){

		this->worms->ZeroForce(); // works
		this->worms->ResetNeighborsList(itime);
	//	this->worms->DisplayNList();
		this->worms->InternalForces(); 
		this->worms->LJForces();
		this->worms->AutoDriveForces();
		this->worms->LandscapeForces();
	//	this->worms->AddConstantForce(0, 1.0f);
		this->worms->Update(); // works
		this->XYZPrint(itime); // works
		this->DisplayErrors();

		this->time += this->simparams->_DT;
	}
	this->fxyz.close();
}

void NWSimulation::XYZPrint(int itime){
	//.. only print when it itime == 0
	if (itime % simparams->_FRAMERATE != 0) return;

	//.. debugging
	//this->worms->DisplayNList();

	//.. count and grab errors
	static int frame = 1;
	printf("\nPrinting frame %i", frame++);
	//this->worms->DislayThetaPhi();
	this->DisplayErrors();
	this->worms->DataDeviceToHost();

	//.. print to ntypes
	static const int ntypes = 4;
	static const char ptypes[ntypes] = { 'A', 'B', 'C', 'D' };
	static const int N = params->_NPARTICLES;
	this->fxyz << N + 4 << std::endl;
	this->fxyz << "Comment line" << std::endl;
	for (int i = 0; i < params->_NPARTICLES; i++){
		int t = (i / params->_NP) % ntypes;
		this->fxyz << ptypes[t] << " " 
				   << worms->r[i] << " " 
				   << worms->r[i + N] << " "
				   << worms->r[i + 2*N] << std::endl;
	}
	this->fxyz << "E " << 0 << " " << 0 << " 0 " << std::endl;
	this->fxyz << "E " << simparams->_XBOX << " " << 0 << " 0 " << std::endl;
	this->fxyz << "E " << 0 << " " << simparams->_YBOX << " 0 " << std::endl;
	this->fxyz << "E " << simparams->_XBOX << " " << simparams->_YBOX << " 0 " << std::endl;
}

void NWSimulation::DisplayErrors(){
	this->rng->DisplayErrors();
	this->worms->DisplayErrors();
}

#endif 