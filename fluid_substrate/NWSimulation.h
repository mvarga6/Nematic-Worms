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
#include "NWXYZIO.h"
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

	//.. outputfile
	std::string outputfile;

	//.. time
	float time;

	//.. clocking
	clock_t timer;

public:
	NWSimulation(int argc, char *argv[]);
	~NWSimulation();
	void Run();

private:
	void XYZPrint(int itime);
	void DisplayErrors();

};
//-------------------------------------------------------------------------------------------
//		IMPLEMENTATION HERE
//-------------------------------------------------------------------------------------------
NWSimulation::NWSimulation(int argc, char *argv[]){

	//.. clear everything on GPU
	//cudaDeviceReset();

	//.. setup parameters (should be done with cmdline input)
	this->params = new WormsParameters();
	Init(this->params, argc, argv);

	//.. setup simulation parameters
	this->simparams = new SimulationParameters();
	Init(this->simparams, argc, argv, this->outputfile);

	//.. show parameters on device
	CheckParametersOnDevice <<< 1, 1 >>>();
	ErrorHandler(cudaDeviceSynchronize());

	//.. setup random number generator
	this->rng = new GRNG(3 * params->_NPARTICLES, 0.0f, 1.0f);

	//.. define the worms
	this->worms = new Worms();

	//.. initial worms object
	this->time = 0.0f;
	this->worms->Init(this->rng, this->params, this->simparams, true, 512);

	//.. outputfile
	this->fxyz.open(this->outputfile.c_str());
	if (!this->fxyz.is_open())
		printf("\n***\nError opening output file: %s!\n***", this->outputfile.c_str());
	else
		printf("\nWriting data to outfile: %s", this->outputfile.c_str());
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
	
	//.. check for errors before starting
	this->DisplayErrors();

	//.. start timer clock
	this->timer = clock();

	//.. MAIN SIMULATION LOOP
	for (int itime = 0; itime < this->simparams->_NSTEPS; itime++){
		
		//.. setup neighbors for iteration
		this->worms->ResetNeighborsList(itime);

		//.. inner loop for high frequency potentials
		for (int jtime = 0; jtime < this->simparams->_NSTEPS_INNER; jtime++){
			this->worms->ZeroForce();
			this->worms->InternalForces();
			this->worms->BendingForces();
			this->worms->XLinkerForces();
			this->worms->LJForces();
			this->worms->QuickUpdate((float)this->simparams->_NSTEPS_INNER);
		}

		//.. finish time set with slow potential forces
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

	//.. count and grab errors
	static int frame = 1;
	this->DisplayErrors();

	//.. pull data from GPU
	this->worms->DataDeviceToHost();
	this->worms->ColorXLinked();

	//.. print to ntypes
	static const int ntypes = 4;
	static const char ptypes[ntypes] = { 'A', 'B', 'C', 'D' };
	static const int N = params->_NPARTICLES;

	int nBlownUp = 0;
	this->fxyz << N + 4 << std::endl;
	this->fxyz << nw::util::xyz::makeParameterLine(this->params, this->simparams, __NW_VERSION__);
	for (int i = 0; i < params->_NPARTICLES; i++){
		//int w = i / params->_NP;
		//int t = w % ntypes;
		float x = worms->r[i], y = worms->r[i + N], z = worms->r[i + 2 * N];
		char c = worms->c[i];
		if (abs(z) > 100.0f) nBlownUp++;
		this->fxyz << c << " " << x << " " << y << " " << z << std::endl;
	}
	this->fxyz << "E " << 0 << " " << 0 << " 0 " << std::endl;
	this->fxyz << "E " << simparams->_XBOX << " " << 0 << " 0 " << std::endl;
	this->fxyz << "E " << 0 << " " << simparams->_YBOX << " 0 " << std::endl;
	this->fxyz << "E " << simparams->_XBOX << " " << simparams->_YBOX << " 0 " << std::endl;

	//.. report blown up particles
	if (nBlownUp > 0) printf("\n%i particles blown up", nBlownUp);
	if (nBlownUp >= _LOSS_TOLERANCE) abort();

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