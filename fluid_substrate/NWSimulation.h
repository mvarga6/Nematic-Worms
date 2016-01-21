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

	//.. clocking
	clock_t timer;

public:
	NWSimulation();
	~NWSimulation();
	void Run();

private:
	void XYZPrint(int itime);
	void DisplayErrors();

};
//-------------------------------------------------------------------------------------------
//		IMPLEMENTATION HERE
//-------------------------------------------------------------------------------------------
NWSimulation::NWSimulation(){

	//.. clear everything on GPU
	cudaDeviceReset();

	//.. setup parameters (should be done with cmdline input)
	this->params = new WormsParameters();
	this->params->_XDIM = 5;
	this->params->_YDIM = 40;
	this->params->_NP = 20;
	this->params->_NWORMS = params->_XDIM * params->_YDIM;
	this->params->_NPARTICLES = params->_NP * params->_NWORMS;
	this->params->_LISTSETGAP = 50;
	this->params->_NMAX = 128;
	this->params->_EPSILON = 0.25f;
	this->params->_SIGMA = 1.0f;
	this->params->_DRIVE = 0.5f;
	this->params->_K1 = 57.146f / 2.0f;
	this->params->_K2 = 10.0f * params->_K1;
	this->params->_K3 = 2.0f * params->_K2 / 3.0f;
	this->params->_Ka = 5.0f;
	this->params->_L1 = 0.80000f;
	this->params->_L2 = 1.60000f;
	this->params->_L3 = 2.40000f;
	this->params->_KBT = 0.05f;
	this->params->_GAMMA = 2.0f;
	this->params->_DAMP = 3.0f;
	this->params->_SIGMA6 = powf(params->_SIGMA, 6.0f);
	this->params->_2SIGMA6 = params->_SIGMA6 * 2.0f;
	this->params->_LJ_AMP = 24.0f * params->_EPSILON * params->_SIGMA6;
	this->params->_RMIN = _216 * params->_SIGMA;
	this->params->_R2MIN = params->_RMIN * params->_RMIN;
	this->params->_RCUT = 2.5f * params->_SIGMA; // params->_RMIN; //
	this->params->_R2CUT = params->_RCUT * params->_RCUT;
	this->params->_BUFFER = 0.20f;
	this->params->_LANDSCALE = 1.0f;

	//.. setup simulation parameters
	this->simparams = new SimulationParameters();
	this->simparams->_DT = 0.05f;
	this->simparams->_FRAMERATE = 1000;
	this->simparams->_FRAMESPERFILE = 200;
	this->simparams->_NSTEPS = 100000;
	this->simparams->_XBOX = 100.0f;
	this->simparams->_YBOX = 80.0f;
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
//-------------------------------------------------------------------------------------------
NWSimulation::~NWSimulation(){
	delete this->worms;
	delete this->rng;
	delete this->params;
	delete this->simparams;
}
//-------------------------------------------------------------------------------------------
void NWSimulation::Run(){
	this->worms->Init(this->rng, this->params, this->simparams, true, 512);
	this->DisplayErrors();
	this->fxyz.open("output//3d_test10.xyz");
	const int innerSteps = 25;

	//this->XYZPrint(0);
	this->timer = clock();
	for (int itime = 0; itime < this->simparams->_NSTEPS; itime++){

		//this->worms->ZeroForce();
		this->worms->ResetNeighborsList(itime);
		for (int jtime = 0; jtime < innerSteps; jtime++){
			this->worms->ZeroForce();
			this->worms->InternalForces();
			this->worms->BendingForces();
			this->worms->LJForces();
			this->worms->QuickUpdate(25.0f);
		}
		this->worms->ZeroForce();
		this->worms->AutoDriveForces();
		this->worms->LandscapeForces();
		this->worms->SlowUpdate();
		this->XYZPrint(itime);
		this->worms->DisplayClocks(itime);
		this->DisplayErrors();

		this->time += this->simparams->_DT;
	}
	this->fxyz.close();
}
//-------------------------------------------------------------------------------------------
void NWSimulation::XYZPrint(int itime){
	//.. only print when it itime == 0
	if (itime % simparams->_FRAMERATE != 0) return;

	//.. debugging
	//this->worms->DisplayNList();

	//.. count and grab errors
	static int frame = 1;
	//this->worms->DislayThetaPhi();
	this->DisplayErrors();
	this->worms->DataDeviceToHost();

	//.. print to ntypes
	static const int ntypes = 4;
	static const char ptypes[ntypes] = { 'A', 'B', 'C', 'D' };
	static const int N = params->_NPARTICLES;

	int nBlownUp = 0;
	this->fxyz << N + 4 << std::endl;
	this->fxyz << "Comment line" << std::endl;
	for (int i = 0; i < params->_NPARTICLES; i++){
		int w = i / params->_NP;
		int t = w % ntypes;
		float x = worms->r[i], y = worms->r[i + N], z = worms->r[i + 2 * N];
		if (abs(z) > 100.0f) nBlownUp++;
		this->fxyz << ptypes[t] << " " 
				   << x << " " << y << " " << z << std::endl;
	}
	this->fxyz << "E " << 0 << " " << 0 << " 0 " << std::endl;
	this->fxyz << "E " << simparams->_XBOX << " " << 0 << " 0 " << std::endl;
	this->fxyz << "E " << 0 << " " << simparams->_YBOX << " 0 " << std::endl;
	this->fxyz << "E " << simparams->_XBOX << " " << simparams->_YBOX << " 0 " << std::endl;

	//.. report blown up particles
	if (nBlownUp > 0) printf("\n%i particles blown up", nBlownUp);
	if (nBlownUp == params->_NPARTICLES) abort();

	//.. clocking
	clock_t now = clock();
	double frame_t = (now - this->timer) / (double)(CLOCKS_PER_SEC * 60.0);
	double frames_left = double(this->simparams->_NSTEPS / this->simparams->_FRAMERATE)
		- (double)(frame - 1);
	double mins_left = frame_t * frames_left;
	double hours_left = mins_left / 60.0;
	int hours = int(hours_left);
	int mins = int((hours_left - (double)hours) * 60);
	printf("\nframe %i: %i hrs %i mins remaining", frame++, hours, mins);
	this->timer = clock();
}
//-------------------------------------------------------------------------------------------
void NWSimulation::DisplayErrors(){
	this->rng->DisplayErrors();
	this->worms->DisplayErrors();
}
//-------------------------------------------------------------------------------------------
#endif 