
#ifndef __WORMS_H__
#define __WORMS_H__
#include "curand_kernel.h"
#include "NWmain.h"
#include "NWWormsKernels.h"
#include "NWParams.h"
#include "NWRandomNumberGenerator.h"
#include "NWWormsParameters.h"
#include "NWSimulationParameters.h"
#include "NWBoundaryFunctions.h"
#include "NWErrorHandler.h"
// 2D
class ForceXchanger;
// -----------------------------------------------------------------------------------------
//	This class defines a set of active flexible worms.
// -----------------------------------------------------------------------------------------
class Worms {	

	//.. new data structure
	float * dev_r;
	float * dev_v;
	float * dev_f;
	float * dev_f_old;
	float * dev_thphi;

	//.. neighbors list within worms
	int * dev_nlist;
	int * dev_xlink;
	int * dev_xcount;

	//.. for pitched memory
	bool pitched_memory;
	size_t rpitch;
	size_t vpitch;
	size_t fpitch;
	size_t fopitch;
	size_t tppitch;
	size_t nlpitch;

	size_t rshift;
	size_t vshift;
	size_t fshift;
	size_t foshift;
	size_t tpshift;
	size_t nlshift;

	//.. pitch memory heights
	size_t height3;
	size_t height2;
	size_t heightNMAX;

	//.. contains errors
	std::vector<cudaError_t> errorState;

	//.. random number generator
	GRNG * rng;

	//.. Parameters for the worms
	WormsParameters * parameters;

	//.. parameters of the environment
	SimulationParameters * envirn;

	//.. block-thread structure
	int Threads_Per_Block;
	int Blocks_Per_Kernel;

	//.. allocation sizes
	size_t nparticles_int_alloc;
	size_t nparticles_float_alloc;

	//.. clocking the function calls
	const int clock_rate;
	std::clock_t LJ_clock,
				 Noise_clock,
				 Internal_clock,
				 Drive_clock,
				 ThPhi_clock,
				 Nlist_clock,
				 Land_clock,
				 Update_clock,
				 DataTrans_clock;

public:
	//.. on host (so print on host is easier)
	float * r;
	char * c;

	//.. construct with block-thread structure
	//Worms(int BPK, int TPB, GRNG &RNG, WormsParameters &wormsParameters);
	Worms(int clockingRate);
	~Worms();

	//.. user access to running simulation
	void Init(GRNG *, WormsParameters *, SimulationParameters *, bool, int);
	void CustomInit(float *headX, float *headY, float *wormAngle);
	void InternalForces();
	void BendingForces();
	void NoiseForces();
	void LJForces();
	void AutoDriveForces(int itime, int istart);
	void LandscapeForces();
	void AddConstantForce(int dim, float force);
	void XLinkerForces(int itime, float xtargetPercent);
	void SlowUpdate();
	void QuickUpdate();
	void CalculateThetaPhi();
	void DataDeviceToHost();
	void DataHostToDevice();
	void ResetNeighborsList();
	void ResetNeighborsList(int itime);
	void ColorXLinked(); // pulls dev_xlink onto host and makes char list
	void ClearAll();
	void ZeroForce();

	//.. for Debugging
	void DisplayNList();
	void DislayThetaPhi();
	void DisplayErrors();
	void DisplayClocks(int itime);

	//.. so force exchanger can do what i needs
	friend class ForceXchanger;

private:
	//.. methods for internal use
	void AllocateHostMemory();
	void FreeHostMemory();
	void AllocateGPUMemory();
	void FreeGPUMemory();
	void ZeroHost();
	void ZeroGPU();
	void ZeroClocks();

	void AllocateGPUMemory_Pitched();
	void DataDeviceToHost_Pitched();
	void DataHostToDevice_Pitched();
	void ZeroGPU_Pitched();

	void DistributeWormsOnHost();
	void PlaceWormExplicit(int wormId, float headX, float headY, float wormAngle);
	void AdjustDistribute(float target_percent);
	void CheckSuccess(cudaError_t err);
	void FigureBlockThreadStructure(int threadsPerBlock);
};
// ------------------------------------------------------------------------------------------
//	PUBLIC METHODS
// ------------------------------------------------------------------------------------------
Worms::Worms(int clockingRate = -1) : 
r(NULL), dev_r(NULL), dev_v(NULL), dev_f(NULL), 
dev_f_old(NULL), dev_thphi(NULL), clock_rate(clockingRate) {
	DEBUG_MESSAGE("Constructor");
	//.. do necessities
	srand(time(NULL));
}
//-------------------------------------------------------------------------------------------
Worms::~Worms(){
	this->ClearAll();
}
//-------------------------------------------------------------------------------------------
void Worms::ClearAll(){

	//.. host
	this->FreeHostMemory();
	this->r = NULL;
	this->c = NULL;

	//.. device
	this->FreeGPUMemory();
	this->dev_r = this->dev_v = NULL;
	this->dev_f = this->dev_f_old = NULL;
	this->dev_xcount = NULL;
	this->dev_xlink = NULL;
	this->dev_thphi = NULL;
	this->rng = NULL;
	this->parameters = NULL;
	this->envirn = NULL;
}
//-------------------------------------------------------------------------------------------
/*	Initializes the worms to run using a /p gaussianRandomNumberGenerator
*	with the parameters specified by /p wormsParameters and utilizeding 
*	an envirnment specified by /p simParameters.  If /p threadsPerBlock
*	is specified, the it will be used, otherwise default is 256.  The init
*	does the following things:
*		1.	Calculates the thread/block structure
*		2.	Allocates and zeros memory on host
*		3.	Allocates and zeros memory on GPU
*		4.	Distributes worms on host
*		5.	Moves worm data to GPU
*		6.	Calculates theta using GPU
*		7.	Sets the inital neighbors lists using GPU
*/
void Worms::Init(GRNG * gaussianRandomNumberGenerator,
				WormsParameters * wormsParameters,
				SimulationParameters * simParameters,
				bool usePitchedMemory = false,
				int threadsPerBlock = 256){
	DEBUG_MESSAGE("Init");
	this->pitched_memory = usePitchedMemory;
	this->rng = gaussianRandomNumberGenerator;
	this->parameters = wormsParameters;
	this->envirn = simParameters;
	this->FigureBlockThreadStructure(threadsPerBlock);
	this->AllocateHostMemory();
	this->ZeroHost();
	this->ZeroClocks();
	this->DistributeWormsOnHost();
	this->AdjustDistribute(0.5f);

	if (usePitchedMemory){
		this->AllocateGPUMemory_Pitched();
		this->ZeroGPU_Pitched();
	}
	else {
		this->AllocateGPUMemory();
		this->ZeroGPU();
	}

	this->DataHostToDevice();
	this->CalculateThetaPhi();
	this->ResetNeighborsList();
	ErrorHandler(cudaDeviceSynchronize());
}
//-------------------------------------------------------------------------------------------
void Worms::CustomInit(float *headX, float *headY, float *wormAngle){
	this->AllocateHostMemory();
	this->ZeroHost();
	this->AllocateGPUMemory();
	this->ZeroGPU();

	//.. Custom distribute here for all worms
	for (int i = 0; i < parameters->_NWORMS; i++)
		this->PlaceWormExplicit(i, headX[i], headY[i], wormAngle[i]);

	this->DataHostToDevice();
	ErrorHandler(cudaDeviceSynchronize());
}
//-------------------------------------------------------------------------------------------
void Worms::InternalForces(){
	DEBUG_MESSAGE("InternalForces");
	std::clock_t b4 = std::clock();
	float noise = sqrtf(2.0f * parameters->_GAMMA * parameters->_KBT / envirn->_DT);
	int N = _D_ * this->parameters->_NPARTICLES;
	float * rng_ptr = this->rng->Get(N);
	InterForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift,
		this->dev_v, this->vshift,
		this->dev_r, this->rshift,
		rng_ptr, noise
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	this->Internal_clock += std::clock() - b4;
}
//-------------------------------------------------------------------------------------------
void Worms::BendingForces(){
	DEBUG_MESSAGE("BendingForces");
	int Blocks = 1 + (this->parameters->_NWORMS / this->Threads_Per_Block);
	BondBendingForces <<< Blocks, this->Threads_Per_Block >>>
	(
			this->dev_f, this->fshift,
			this->dev_r, this->rshift
	);
}
//-------------------------------------------------------------------------------------------
void Worms::NoiseForces(){
	DEBUG_MESSAGE("NoiseForces");
	std::clock_t b4 = std::clock();
	dim3 gridStruct(this->Blocks_Per_Kernel, _D_);
	dim3 blockStruct(this->Threads_Per_Block);
	int N = _D_ * this->parameters->_NPARTICLES;
	float noise = sqrtf(2.0f * parameters->_GAMMA * parameters->_KBT / envirn->_DT);
	NoiseKernel <<< gridStruct, blockStruct >>>
	(
		this->dev_f, this->fshift,
		this->rng->Generate(N), noise
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	this->Noise_clock += std::clock() - b4;
}
//-------------------------------------------------------------------------------------------
void Worms::LJForces(){
	DEBUG_MESSAGE("LJForces");
	std::clock_t b4 = std::clock();
	LennardJonesNListKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift, 
		this->dev_r, this->rshift, 
		this->dev_nlist, this->nlshift
	);
	/*dim3 threadStruct(16, 16);
	dim3 numBlocks((this->parameters->_NPARTICLES / threadStruct.x) + 1, 
				   (this->parameters->_NMAX / threadStruct.y) + 1);
	FastLennardJonesNListKernel <<< numBlocks, threadStruct >>>
	(
		this->dev_f, this->fshift,
		this->dev_r, this->rshift,
		this->dev_nlist, this->nlshift
	);*/
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	this->LJ_clock += std::clock() - b4;
}
//-------------------------------------------------------------------------------------------
void Worms::AutoDriveForces(int itime, int istart = 0){
	DEBUG_MESSAGE("AutoDriveForces");
	if (itime < istart) return;
	this->CalculateThetaPhi();
	std::clock_t b4 = std::clock();
	DriveForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift, 
		this->dev_thphi, this->tpshift
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	this->Drive_clock += std::clock() - b4;
	
}
//-------------------------------------------------------------------------------------------
void Worms::LandscapeForces(){
	DEBUG_MESSAGE("LandscapeForces");
	std::clock_t b4 = std::clock();
	WormsLandscapeKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift,
		this->dev_r, this->rshift
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	this->Land_clock += std::clock() - b4;
}
//-------------------------------------------------------------------------------------------
void Worms::XLinkerForces(int itime, float xtargetPercent = 0.0f){
	DEBUG_MESSAGE("XLinkerForces");

	//.. only process if needed
	if (itime < this->parameters->_XSTART) return;
	if (xtargetPercent <= 0.0f) return;
	if (xtargetPercent > 1.0f) return;

	//.. grab needed parameters
	const float crossLinkDensityTarget = xtargetPercent;
	const int N = this->parameters->_NPARTICLES;

	//.. break a 10% percentage of cross links
	float * uni_rn = this->rng->Get(N, true);
	XLinkerBreakKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_xlink, uni_rn, 0.00001f
	);

	//.. count linkages
	int currentNumber;
	XLinkerCountKernel<<<1, 1>>>(this->dev_xlink, this->dev_xcount);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	
	//.. calculate current density
	CheckSuccess(cudaMemcpy(&currentNumber, this->dev_xcount, sizeof(int), cudaMemcpyDeviceToHost));
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	const float currentDensity = float(currentNumber) / float(N);
	const float offsetDensity = crossLinkDensityTarget - currentDensity;

	//.. adjust linkages to target percentage
	float * rng_ptr = this->rng->Get(N, true);
	float linkCutoff = this->parameters->_Lx;
	XLinkerUpdateKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_r, this->rshift,
		this->dev_xlink,
		this->dev_nlist, this->nlshift,
		rng_ptr,
		offsetDensity, linkCutoff
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());

	//.. apply forces from linkages to linker particle
	XLinkerForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift,
		this->dev_r, this->rshift,
		this->dev_xlink, true
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	
	//.. apply forces from linkages to linked particle 2
	XLinkerForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
		(
		this->dev_f, this->fshift,
		this->dev_r, this->rshift,
		this->dev_xlink, false
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	DEBUG_MESSAGE("XLinkerForces_forces2");
}
//-------------------------------------------------------------------------------------------
void Worms::SlowUpdate(){
	DEBUG_MESSAGE("SlowUpdate");

	/*UpdateSystemKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fpitch, 
		this->dev_f_old, this->fopitch, 
		this->dev_v, this->vpitch, 
		this->dev_r, this->rpitch
	);*/
	std::clock_t b4 = std::clock();
	dim3 gridStruct(this->Blocks_Per_Kernel, _D_);
	dim3 blockStruct(this->Threads_Per_Block);
	FastUpdateKernel <<< gridStruct, blockStruct >>>
	(
		this->dev_f, this->fshift,
		this->dev_f_old, this->foshift,
		this->dev_v, this->vshift,
		this->dev_r, this->rshift,
		this->envirn->_DT
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	this->Update_clock += std::clock() - b4;
}
//-------------------------------------------------------------------------------------------
void Worms::QuickUpdate(){
	DEBUG_MESSAGE("QuickUpdate");

	/*UpdateSystemKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
	this->dev_f, this->fpitch,
	this->dev_f_old, this->fopitch,
	this->dev_v, this->vpitch,
	this->dev_r, this->rpitch
	);*/

	std::clock_t b4 = std::clock();
	const float increaseRatio = (float)this->envirn->_NSTEPS_INNER;
	dim3 gridStruct(this->Blocks_Per_Kernel, _D_);
	dim3 blockStruct(this->Threads_Per_Block);
	FastUpdateKernel <<< gridStruct, blockStruct >>>
	(
		this->dev_f, this->fshift,
		this->dev_f_old, this->foshift,
		this->dev_v, this->vshift,
		this->dev_r, this->rshift,
		this->envirn->_DT / increaseRatio
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	this->Update_clock += std::clock() - b4;
}
//-------------------------------------------------------------------------------------------
void Worms::CalculateThetaPhi(){
	DEBUG_MESSAGE("CalculateThetaPhi");
	std::clock_t b4 = std::clock();
	CalculateThetaKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_r, this->rshift, 
		this->dev_thphi, this->tpshift
	);
	ErrorHandler(cudaDeviceSynchronize());
	FinishCalculateThetaKernel <<< (this->parameters->_NWORMS / this->Threads_Per_Block) + 1, this->Threads_Per_Block >>>
	(
		this->dev_thphi, this->tpshift
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	this->ThPhi_clock += std::clock() - b4;
}
//-------------------------------------------------------------------------------------------
void Worms::DataDeviceToHost(){
	DEBUG_MESSAGE("DataDeviceToHost");

	std::clock_t b4 = std::clock();
	/*CheckSuccess(cudaMemcpy2D(this->r,
		this->nparticles_float_alloc,
		this->dev_r,
		this->rpitch,
		this->nparticles_float_alloc,
		this->height3,
		cudaMemcpyDeviceToHost));
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());*/
	if (this->pitched_memory) 
		this->DataDeviceToHost_Pitched();
	else {
		CheckSuccess(cudaMemcpy(this->r, 
			this->dev_r, 
			_D_ * this->nparticles_float_alloc,
			cudaMemcpyDeviceToHost));
		ErrorHandler(cudaDeviceSynchronize());
		ErrorHandler(cudaGetLastError());
	}
	this->DataTrans_clock += std::clock() - b4;
}
//-------------------------------------------------------------------------------------------
void Worms::DataHostToDevice(){
	DEBUG_MESSAGE("DataHostToDevice");

	if (this->pitched_memory)
		this->DataHostToDevice_Pitched();
	else {
		CheckSuccess(cudaMemcpy(this->dev_r, this->r, _D_ * this->nparticles_float_alloc, cudaMemcpyHostToDevice));
		ErrorHandler(cudaDeviceSynchronize());
		//ErrorHandler(cudaGetLastError());
	}

	/*CheckSuccess(cudaMemcpy2D(this->dev_r,
		this->rpitch,
		this->r,
		this->nparticles_float_alloc,
		this->nparticles_float_alloc,
		this->height3,
		cudaMemcpyHostToDevice));
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());*/
}
//-------------------------------------------------------------------------------------------
//.. sets neighbors list between worm-worm and worm-fluid
void Worms::ResetNeighborsList(){
	DEBUG_MESSAGE("ResetNeighborsList");
	std::clock_t b4 = std::clock();

	//.. reset memory to -1
	if (pitched_memory) {
		CheckSuccess(cudaMemset2D(this->dev_nlist,
			this->nlpitch,
			-1,
			this->nparticles_int_alloc,
			this->heightNMAX));
	}
	else {
		CheckSuccess(cudaMemset(this->dev_nlist, -1, 
			this->parameters->_NMAX * this->nparticles_int_alloc));

	}
	ErrorHandler(cudaDeviceSynchronize());

	//.. assign neighbors
	SetNeighborList_N2Kernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_r, this->rshift,
		this->dev_nlist, this->nlshift
	);

	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
	this->Nlist_clock += std::clock() - b4;
}
//-------------------------------------------------------------------------------------------
void Worms::ResetNeighborsList(int itime){
	if (itime % parameters->_LISTSETGAP == 0)
		this->ResetNeighborsList();
}
//-------------------------------------------------------------------------------------------
void Worms::DislayThetaPhi(){

	float * thphi = new float[2 * this->parameters->_NPARTICLES];

	if (pitched_memory) {
		CheckSuccess(cudaMemcpy2D(thphi,
			this->nparticles_float_alloc,
			this->dev_thphi,
			this->tppitch,
			this->nparticles_float_alloc,
			this->height2,
			cudaMemcpyDeviceToHost));
	}
	else {
		CheckSuccess(cudaMemcpy(thphi,
			this->dev_thphi,
			2 * this->nparticles_float_alloc,
			cudaMemcpyDeviceToHost));
	}

	ErrorHandler(cudaDeviceSynchronize());
	for (int i = 0; i < this->parameters->_NPARTICLES; i++){
		/*if (isnan(thphi[i]))
			std::cout << "thphi[" << i << "] = NaN" << std::endl;
		if (isnan(thphi[i + this->parameters->_NPARTICLES])) 
			std::cout << "thphi[" << i << "] = NaN" << std::endl;*/
		std::cout << "\nth = " << thphi[i] << "\tphi = " << thphi[i + this->parameters->_NPARTICLES];
	}
	delete[] thphi;
}
//-------------------------------------------------------------------------------------------
void Worms::DisplayNList(){

	size_t size = this->parameters->_NPARTICLES * this->parameters->_NMAX;
	int * nlist = new int[size];

	if (pitched_memory) {
		CheckSuccess(cudaMemcpy2D(nlist,
			this->nparticles_int_alloc,
			this->dev_nlist,
			this->nlpitch,
			this->nparticles_int_alloc,
			this->heightNMAX,
			cudaMemcpyDeviceToHost));
	}
	else {
		CheckSuccess(cudaMemcpy(nlist, 
			this->dev_nlist, 
			size * sizeof(int), 
			cudaMemcpyDeviceToHost));
	}
	ErrorHandler(cudaDeviceSynchronize());

	for (int i = 0; i < parameters->_NPARTICLES; i++){
		//printf("\nParticle %i\n", i);
		for (int n = 0; n < parameters->_NMAX; n++){
			//printf("\t%i == %i", n, nlist[i + n*this->parameters->_NPARTICLES]);
			int p = nlist[i + n*this->parameters->_NPARTICLES];
			if (p != -1) printf("\nParticle %i:\t%i", i, p);
		}
	}
	delete[] nlist;
}
//-------------------------------------------------------------------------------------------
void Worms::DisplayErrors(){
	if (this->errorState.size() > 0){
		printf("\nWORM ERRORS:\n");
		for (int i = 0; i < this->errorState.size(); i++) {
			//printf("%i -- %s\n", i, cudaGetErrorString(this->errorState[i]));
			ErrorHandler(this->errorState[i]);
		}
		this->errorState.empty();
	}
	//this->DislayThetaPhi();
}
//-------------------------------------------------------------------------------------------
void Worms::DisplayClocks(int itime){

	if (this->clock_rate == -1) return;
	if (itime % this->clock_rate != 0) return;

	printf("\n\nWorms Performance");
	printf("\nLJ\t\t%d", this->LJ_clock / this->clock_rate);
	printf("\nNoise\t\t%d", this->Noise_clock / this->clock_rate);
	printf("\nInternal\t%d", this->Internal_clock / this->clock_rate);
	printf("\nDrive\t\t%d", this->Drive_clock / this->clock_rate);
	printf("\nDataTrans\t%d", this->DataTrans_clock / this->clock_rate);
	printf("\nThPhi\t\t%d", this->ThPhi_clock / this->clock_rate);
	printf("\nNlist\t\t%d", this->Nlist_clock / this->clock_rate);
	printf("\nLand\t\t%d", this->Land_clock / this->clock_rate);
	printf("\nUpdate\t\t%d\n", this->Update_clock / this->clock_rate);

	this->ZeroClocks();
}
//-------------------------------------------------------------------------------------------
void Worms::ZeroForce(){
	DEBUG_MESSAGE("ZeroForces");
	
	if (this->pitched_memory) {
		CheckSuccess(cudaMemset2D(this->dev_f,
			this->fpitch,
			0,
			this->parameters->_NPARTICLES*sizeof(float),
			_D_));
	}
	else {
		CheckSuccess(cudaMemset(this->dev_f, 0, _D_ * this->nparticles_float_alloc));
	}
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}
//-------------------------------------------------------------------------------------------
void Worms::AddConstantForce(int dim, float force){
	DEBUG_MESSAGE("AddConstantForce");

	AddForce <<<  this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fpitch,
		dim, force
	);
	ErrorHandler(cudaDeviceSynchronize());
}
//-------------------------------------------------------------------------------------------
void Worms::ColorXLinked(){
	const int N = this->parameters->_NPARTICLES;
	int * xlink = new int[N]; // create host copy of dev_xlink
	CheckSuccess(cudaMemcpy(xlink, this->dev_xlink, this->nparticles_int_alloc, cudaMemcpyDeviceToHost));
	for (int i = 0; i < N; i++){ // find linked particles
		if (xlink[i] == -1) this->c[i] = 'A'; // type A if not linked
		else this->c[i] = 'B'; // type B if linked
	}
	delete[] xlink; // delete local copy
}
// ------------------------------------------------------------------------------------------
//	PRIVATE METHODS
// ------------------------------------------------------------------------------------------
void Worms::AllocateHostMemory(){
	DEBUG_MESSAGE("AllocateHostMemory");
	this->r = new float[_D_*this->parameters->_NPARTICLES];
	this->c = new char[this->parameters->_NPARTICLES];
}
//-------------------------------------------------------------------------------------------
void Worms::FreeHostMemory(){
	DEBUG_MESSAGE("FreeHostMemory");
	delete[] this->r;
	delete[] this->c;
}
//-------------------------------------------------------------------------------------------
void Worms::AllocateGPUMemory(){
	DEBUG_MESSAGE("AllocateGPUMemory");

	printf("\nUsing linear memory:\t");
	
	//.. calculate and assign shifts as normal
	this->fshift = this->parameters->_NPARTICLES;
	this->foshift = this->parameters->_NPARTICLES;
	this->vshift = this->parameters->_NPARTICLES;
	this->rshift = this->parameters->_NPARTICLES;
	this->tpshift = this->parameters->_NPARTICLES;
	this->nlshift = this->parameters->_NPARTICLES;

	//.. allocate linear memory
	CheckSuccess(cudaMalloc((void**)&this->dev_r, _D_ * this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_v, _D_ * this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_f, _D_ * this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_f_old, _D_ * this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_nlist, this->parameters->_NMAX * this->nparticles_int_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_thphi, 2 * this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_xlink, this->nparticles_int_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_xcount, sizeof(int)));

	ErrorHandler(cudaDeviceSynchronize());
	printf("Allocated");
}
//-------------------------------------------------------------------------------------------
void Worms::AllocateGPUMemory_Pitched(){
	DEBUG_MESSAGE("AllocateGPUMemory_Pitched");

	printf("\nUsing pitched memory:\t");

	this->height3 = _D_;
	this->height2 = 2;
	this->heightNMAX = this->parameters->_NMAX;
	size_t widthN = this->nparticles_float_alloc;

	CheckSuccess(cudaMallocPitch(&this->dev_r, 
								&this->rpitch, 
								widthN,
								_D_));

	CheckSuccess(cudaMallocPitch(&this->dev_v,
								&this->vpitch,
								widthN,
								_D_));

	CheckSuccess(cudaMallocPitch(&this->dev_f,
								&this->fpitch,
								widthN,
								_D_));

	CheckSuccess(cudaMallocPitch(&this->dev_f_old,
								&this->fopitch,
								widthN,
								_D_));

	CheckSuccess(cudaMallocPitch(&this->dev_nlist,
								&this->nlpitch,
								this->nparticles_int_alloc,
								this->heightNMAX));

	CheckSuccess(cudaMallocPitch(&this->dev_thphi,
								&this->tppitch,
								widthN,
								this->height2));

	CheckSuccess(cudaMalloc((void**)&this->dev_xlink, this->nparticles_int_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_xcount, sizeof(int)));

	//.. calculate and assign shifts
	this->fshift = this->fpitch / sizeof(float);
	this->foshift = this->fopitch / sizeof(float);
	this->vshift = this->vpitch / sizeof(float);
	this->rshift = this->rpitch / sizeof(float);
	this->tpshift = this->tppitch / sizeof(float);
	this->nlshift = this->nlpitch / sizeof(int);

	//.. print to user pitches
	printf("\nDevice memory pitches:\n----------------------\nf:\t%i\nf_old:\t%i\nv:\t%i\nr:\t%i\nthephi:\t%i\nnlist:\t%i\n",
		this->fpitch, 
		this->fopitch, 
		this->vpitch, 
		this->rpitch,
		this->tppitch,
		this->nlpitch);

	//.. print to user shifts
	printf("\nDevice memory shifts:\n-----------------------\nf:\t%i\nf_old:\t%i\nv:\t%i\nr:\t%i\nthephi:\t%i\nnlist:\t%i\n",
		this->fshift, 
		this->foshift, 
		this->vshift, 
		this->rshift,
		this->tpshift,
		this->nlshift);

	printf("Allocated\n");
}
//-------------------------------------------------------------------------------------------
void Worms::FreeGPUMemory(){
	DEBUG_MESSAGE("FreeGPUMemory");
	cudaDeviceReset();
}
//-------------------------------------------------------------------------------------------
void Worms::DistributeWormsOnHost(){

	//.. grab parameters
	int nworms = this->parameters->_NWORMS;
	int nparts = this->parameters->_NPARTICLES;
	int np = this->parameters->_NP;
	int xdim = this->parameters->_XDIM;
	int ydim = this->parameters->_YDIM;
	//int zdim = this->parameters->_ZDIM;
	float l1 = this->parameters->_L1;
	float xbox = this->envirn->_XBOX;
	float ybox = this->envirn->_YBOX;

	float *x0 = new float[nworms];
	float *y0 = new float[nworms];
	//float *z0 = new float[nworms];
	int iw = 0;
	//for (int k = 0; k < zdim; k++){
	for (int i = 0; i < xdim; i++){
		for (int j = 0; j < ydim; j++){
			x0[iw] = 0.001 + float(i)*xbox / float(xdim);
			y0[iw] = 0.001 + float(j)*ybox / float(ydim);
			//z0[iw] = float(k) * parameters->_RCUT + 20.0f;
			iw++;
		}
	}
	//}

	// Distribute bodies
	const float s[3] = { l1, 0, 0 }; // always 3d
	for (int i = 0; i < nparts; i++)
	{
		float rn[_D_];
		for_D_ rn[d] = 0.25f*float(rand()) / float(RAND_MAX);
		//float rx = 0.25f*float(rand()) / float(RAND_MAX);
		//float ry = 0.25f*float(rand()) / float(RAND_MAX);
		//float rz = 0.25f*float(rand()) / float(RAND_MAX);
		const int w = i / np;
		const int p = i % np;
		float _r[_D_];
		_r[0] = x0[w] + p * l1 + rn[0];
		_r[1] = y0[w] + rn[1];
		//float z = z0[w];
		MovementBC(_r[0], xbox);
		MovementBC(_r[1], ybox);
		this->r[i + 0*nparts] = _r[0];
		this->r[i + 1*nparts] = _r[1];
		//this->r[i + 2*nparts] = z + rz;
	}

	delete[] x0;
	delete[] y0;
	//delete[] z0;
}
//-------------------------------------------------------------------------------------------
void Worms::AdjustDistribute(float target_percent){

	//.. grab parameters
	const int np = this->parameters->_NP;
	const int nworms = this->parameters->_NWORMS;
	const int N = np * nworms;

	//.. flip orientation for target_percent of worms
	float * save = new float[_D_ * np]; 
	//float * save = new float[np];
	//float * save = new float[np]; 

	for (int w = 0; w < nworms; w++)
	{
		float random = float(rand()) / float(RAND_MAX);
		if (random <= target_percent)
		{
			//.. store position of particles to be flipped
			for (int i = 0; i < np; i++)
			{
				int id = w*np + i;
				for_D_ save[i + d*np] = this->r[id + d*this->rshift];
				//savex[i] = this->r[id];
				//savey[i] = this->r[id + N];
				//savez[i] = this->r[id + 2 * N];
			}

			//.. replace reversed particles
			for (int j = 0; j < np; j++)
			{
				int id = w*np + j;
				for_D_ this->r[id + d*this->rshift] = save[(np - 1 - j) + d*np];
				//this->r[id] = savex[(np - 1 - j)];
				//this->r[id + N] = savey[(np - 1 - j)];
				//this->r[id + 2 * N] = savez[(np - 1) - j];
			}
		}
	}
}
//-------------------------------------------------------------------------------------------
void Worms::PlaceWormExplicit(int wormId, float headX, float headY, float wormAngle){
	
	//.. grab parameters
	float l1 = this->parameters->_L1;
	int np = this->parameters->_NP;
	int nworms = this->parameters->_NWORMS;
	int N = np * nworms;

	if (wormId >= nworms || wormId < 0) return;

	//.. get index of head particle
	int head_id = wormId * np;

	//.. calculate displacement based on angle
	const float dx = l1 * cosf(wormAngle);
	const float dy = l1 * sinf(wormAngle);

	//.. place head particle first
	this->r[head_id] = headX; 
	this->r[head_id + N] = headY;

	//.. place the rest at the previous pos plus dx,dy
	for (int id = head_id + 1; id < (head_id + np); id++){
		this->r[id] = this->r[(id - 1)] + dx;
		this->r[id + N] = this->r[(id - 1) + N] + dy;
		MovementBC(this->r[id], this->envirn->_XBOX);
		MovementBC(this->r[id + N], this->envirn->_YBOX);
	}
}
//-------------------------------------------------------------------------------------------
void Worms::ZeroHost(){
	for (int i = 0; i < _D_ *this->parameters->_NPARTICLES; i++)
		this->r[i] = 0.0f;
}
//-------------------------------------------------------------------------------------------
void Worms::DataDeviceToHost_Pitched(){
	CheckSuccess(cudaMemcpy2D(this->r,
		this->nparticles_float_alloc,
		this->dev_r,
		this->rpitch,
		this->nparticles_float_alloc,
		_D_,
		cudaMemcpyDeviceToHost));
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}
//-------------------------------------------------------------------------------------------
void Worms::DataHostToDevice_Pitched(){
	CheckSuccess(cudaMemcpy2D(this->dev_r,
		this->rpitch,
		this->r,
		this->nparticles_float_alloc,
		this->nparticles_float_alloc,
		_D_,
		cudaMemcpyHostToDevice));
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}
//-------------------------------------------------------------------------------------------
void Worms::ZeroGPU(){
	DEBUG_MESSAGE("ZeroGPU");

	CheckSuccess(cudaMemset((void**)this->dev_r, 0, _D_ * this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_v, 0, _D_ * this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_f, 0, _D_ * this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_f_old, 0, _D_ * this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_thphi, 0, 2 * this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_nlist, -1, this->parameters->_NMAX * this->nparticles_int_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_xlink, -1, this->nparticles_int_alloc));
	ErrorHandler(cudaDeviceSynchronize());
	printf("\nMemory zeroed");
}
//-------------------------------------------------------------------------------------------
void Worms::ZeroGPU_Pitched(){
	DEBUG_MESSAGE("ZeroGPU_Pitched");

	size_t widthN = this->nparticles_float_alloc;

	CheckSuccess(cudaMemset2D(this->dev_r,
		this->rpitch,
		0,
		widthN,
		_D_));

	CheckSuccess(cudaMemset2D(this->dev_v,
		this->vpitch,
		0,
		widthN,
		_D_));

	CheckSuccess(cudaMemset2D(this->dev_f,
		this->fpitch,
		0,
		widthN,
		_D_));

	CheckSuccess(cudaMemset2D(this->dev_f_old,
		this->fopitch,
		0,
		widthN,
		_D_));

	CheckSuccess(cudaMemset2D(this->dev_thphi,
		this->tppitch,
		0,
		widthN,
		this->height2));

	CheckSuccess(cudaMemset2D(this->dev_nlist,
		this->nlpitch,
		-1,
		this->nparticles_int_alloc,
		this->heightNMAX));

	CheckSuccess(cudaMemset((void**)this->dev_xlink, -1, this->nparticles_int_alloc));
}
//-------------------------------------------------------------------------------------------
void Worms::ZeroClocks(){
	this->LJ_clock = 0.0f;
	this->Land_clock = 0.0f;
	this->Nlist_clock = 0.0f;
	this->Update_clock = 0.0f;
	this->DataTrans_clock = 0.0f;
	this->Drive_clock = 0.0f;
	this->ThPhi_clock = 0.0f;
	this->Noise_clock = 0.0f;
	this->Internal_clock = 0.0f;
}
//-------------------------------------------------------------------------------------------
void Worms::CheckSuccess(cudaError_t err){
	if (err != cudaSuccess) {
		this->errorState.push_back(err);
		ErrorHandler(err);
	}
}
//-------------------------------------------------------------------------------------------
void Worms::FigureBlockThreadStructure(int tpb){
	if (tpb >= 0) this->Threads_Per_Block = tpb;
	else this->Threads_Per_Block = 256; // good default value
	this->Blocks_Per_Kernel = this->parameters->_NPARTICLES / this->Threads_Per_Block + 1; // add one to guarentee enough
	this->nparticles_float_alloc = this->parameters->_NPARTICLES * sizeof(float);
	this->nparticles_int_alloc = this->parameters->_NPARTICLES * sizeof(int);
	printf("\nWorms:\n------\nTotalThreads = %i\nBlocksPerKernel = %i\nThreadsPerBlock = %i", 
		this->parameters->_NPARTICLES, 
		this->Blocks_Per_Kernel, 
		this->Threads_Per_Block);
	printf("\nfloat allocs = %i\nint allocs = %i\n",
		this->nparticles_float_alloc,
		this->nparticles_int_alloc);
}
//-------------------------------------------------------------------------------------------
#endif