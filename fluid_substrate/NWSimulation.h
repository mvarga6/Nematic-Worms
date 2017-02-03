#ifndef __SIMULATION_H__
#define __SIMULATION_H__
// 2D
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
	void ReconsileParameters(SimulationParameters *sP, WormsParameters *wP);

};
//-------------------------------------------------------------------------------------------
//		IMPLEMENTATION HERE
//-------------------------------------------------------------------------------------------
NWSimulation::NWSimulation(int argc, char *argv[]){

	srand(std::time(NULL));

	//.. setup parameters (should be done with cmdline input)
	this->params = new WormsParameters();
	Init(this->params, argc, argv);

	//.. setup simulation parameters
	this->simparams = new SimulationParameters();
	Init(this->simparams, argc, argv, this->outputfile);

	//.. Update parameter sets for consistency
	this->ReconsileParameters(this->simparams, this->params);

	//.. setup random number generator
	this->rng = new GRNG(_D_ * params->_NPARTS_ADJ, 0.0f, 1.0f);

	//.. define the worms
	this->worms = new Worms();

	//.. initial worms object
	this->time = 0.0f;
	this->worms->Init(this->rng, this->params, this->simparams, this->simparams->_LMEM, 512);

	//.. send parameters to device once initialized
	ParametersToDevice(*params);
	ParametersToDevice(*simparams);

	//.. show parameters on device
	CheckParametersOnDevice <<< 1, 1 >>>();
	ErrorHandler(cudaDeviceSynchronize());

	//.. outputfile
	this->fxyz.open(this->outputfile.c_str());
	if (!this->fxyz.is_open())
		printf("\n***\nError opening output file: %s!\n***", this->outputfile.c_str());
	else
		printf("\nWriting data to outfile: %s\n", this->outputfile.c_str());
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
	
	this->XYZPrint(0); 
	//return; // to test init positions only

	//.. grab needed parameters
	const int	nsteps		 = this->simparams->_NSTEPS;
	const int	nsteps_inner = this->simparams->_NSTEPS_INNER;
	const float dt			 = this->simparams->_DT;

	//.. check for errors before starting
	this->DisplayErrors();

	//.. start timer clock
	this->timer = clock();

	//.. MAIN SIMULATION LOOP
	this->worms->ZeroForce();
	float encap_l = 0.5f; int range;
	for (int itime = 0; itime < nsteps; itime++){
		
		//.. flexible encapsilation
		
		if (encap_l > 0.25f){
			encap_l -= 0.000025;
			//printf("\n[ %f ] : Encapsulation bond length", encap_l);
		}

		//.. setup neighbors for iteration
		this->worms->ResetNeighborsList(itime);

		//.. inner loop for high frequency potentials
		for (int jtime = 0; jtime < nsteps_inner; jtime++){
			this->worms->InternalForces(encap_l);
			this->worms->BendingForces();
			this->worms->LJForces();
			this->worms->QuickUpdate(range);
		}

		//.. finish time set with slow potential forces
		this->worms->AutoDriveForces(itime);
		this->worms->SlowUpdate(range);
		this->XYZPrint(itime);
		this->worms->DisplayClocks(itime);
		this->DisplayErrors();

		//.. adjust tickers
		this->time += dt;
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
	//this->worms->ColorXLinked();

	//.. print to ntypes
	const int maxTypes = 5;
	const float ka_ratio = this->params->_Ka_RATIO;
	//const int ntypes = (ka_ratio < 1.0f ? 2 : maxTypes);
	const char ptypes[maxTypes] = { 'A', 'B', 'C', 'D', 'E' };
	const int N = params->_NPARTS_ADJ;
	const int nworms = params->_NWORMS;
	const int np = params->_NP;
	const float Lx = simparams->_BOX[0];
	const float Ly = simparams->_BOX[1];

	int nBlownUp = 0;
	this->fxyz << N + 4 << std::endl;
	this->fxyz << nw::util::xyz::makeParameterLine(this->params, this->simparams, __NW_VERSION__);
	for (int i = 0; i < N; i++){
		const int w = i / np;
		const int n = i % np;
		
		float dx = worms->r[(i+1) + 0 * N] - worms->r[i + 0 * N];
		float dy = worms->r[(i + 1) + 1 * N] - worms->r[i + 1 * N];
		BC_dr(dx, Lx, 0);
		BC_dr(dy, Ly, 1);

		float C;
		if (n < (np - 1)) C = (atan2(dy, dx) + PI)/ (2*PI);
		if (i > params->_NPARTICLES) C = 0;

		// choose 0 or 1,2,3,4 type
		const int t = (w > nworms*ka_ratio ? 0 : w % (maxTypes - 1) + 1);
		float _r[3] = { 0, 0, 0 }; // always 3d
		for_D_ _r[d] = worms->r[i + d*N];

		//float x = worms->r[i], y = worms->r[i + N], z = 0.0f;
		//char c = worms->c[i];
		char c = ptypes[t];
		if (i >= this->params->_NPARTICLES) c = 'F';
		if (isnan(_r[0]) || isinf(_r[0])) nBlownUp++;
		this->fxyz << c << " " << _r[0] << " " << _r[1] << " " << _r[2] << std::endl;
	}
	this->fxyz << "F " << 0 << " " << 0 << " 0 0" << std::endl;
	this->fxyz << "F " << simparams->_XBOX << " " << 0 << " 0 0" << std::endl;
	this->fxyz << "F " << 0 << " " << simparams->_YBOX << " 0 0" << std::endl;
	this->fxyz << "F " << simparams->_XBOX << " " << simparams->_YBOX << " 0 0" << std::endl;

	//.. report blown up particles
	if (nBlownUp > 0) {
		printf("\n[ ERROR ] : %i particles blown up, aborting.", nBlownUp);
		exit(EXIT_FAILURE);
	}

	//.. clocking
	clock_t now = clock();
	double frame_t = (now - this->timer) / (double)(CLOCKS_PER_SEC * 60.0);
	double frames_left = double(this->simparams->_NSTEPS / this->simparams->_FRAMERATE)
		- (double)(frame - 1);
	double mins_left = frame_t * frames_left;
	double hours_left = mins_left / 60.0;
	int hours = int(hours_left);
	int mins = int((hours_left - (double)hours) * 60);
	printf("\rframe %i: %i hrs %i mins remaining", frame++, hours, mins);
	this->timer = clock();
}
//-------------------------------------------------------------------------------------------
void NWSimulation::DisplayErrors(){
	this->rng->DisplayErrors();
	this->worms->DisplayErrors();
}
//-------------------------------------------------------------------------------------------
void NWSimulation::ReconsileParameters(SimulationParameters *sP, WormsParameters *wP){
	printf("\nParameters being adjusted for ... ");

	//.. Adjust the number of particles to reflect a flexible encapsilation (2D-only)
#if _D_ == 2
	if (sP->_FLEX_ENCAPS) {
		printf("\n\nFLEXIBLE ENCAPSILATION\n");
		const int worm_n = wP->_NPARTICLES;
		const float xbox = sP->_BOX[0];
		const float ybox = sP->_BOX[1];
		const float init_encap_l = 0.5f;

		//.. make initial diameter of encaps corner to corner distance to
		//	 ensure all inital positions of worms will fit inside enscapsilation
		const float encap_d = (xbox+ybox) / 4.0f;
		
		//.. determine number of particles to add to fill the circumference
		const int encap_n = int((M_PI * encap_d) / init_encap_l);

		//.. adjusted box size (xbox & ybox == encap diameter + eps)
		//   actual box size with be set to this after partice init
		for_D_ sP->_BOX_ADJ[d] = sP->_BOX[d];
		//sP->_BOX_ADJ[0] = sP->_BOX_ADJ[1] = 2 * encap_d;

		//.. set adjust particle number (N-worms + N-encap)
		wP->_NPARTS_ADJ = wP->_NPARTICLES + encap_n;

		printf("\n\nAdjusted Particle #:\t%i", wP->_NPARTS_ADJ);
	}
#endif

	
}
#endif 