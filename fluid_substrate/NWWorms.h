
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
//#include "NWKNN.h"
// 2D
class ForceXchanger;

//texture<float, 2, cudaReadModeElementType> wTex;

// -----------------------------------------------------------------------------------------
//	This class defines a set of active flexible worms.
// -----------------------------------------------------------------------------------------
class Worms {	

	//.. data structures
	float *dev_r, *dev_v, *dev_f, *dev_f_old, *dev_thphi;

	//.. neighbors list within worms
	int *dev_nlist, *host_nlist, *host_nlist_cntr; // stores id of neighbors
	float *dev_ndist, *host_ndist; // stores distances to neighbor
	int *dev_cell; // stores cell of particle

	float dcell; // width of cell domains
	int *heads; // linked list start markers for each cell
	int *ptz; // linked list linkers
	int ncells, nxcell, nycell;
	
	//.. stuff for KNN
	/*float * dist_dev;
	int	* ind_dev;
	size_t dist_pitch,dist_shift,ind_pitch,ind_shift;
	size_t max_nb_query_traited;
	size_t actual_nb_query_width;
	int K;

	unsigned int size_of_float, size_of_int;
	size_t memory_total, memory_free;*/

	//.. for pitched memory
	bool pitched_memory;
	size_t rpitch,vpitch,fpitch,fopitch,tppitch,nlpitch,ndpitch,cpitch;
	size_t rshift,vshift,fshift,foshift,tpshift,nlshift,ndshift,cshift;

	//.. pitch memory heights
	size_t height3,height2,heightNMAX;

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

public:
	//.. on host (so print on host is easier)
	float * r;
	char * c;

	//.. construct with block-thread structure
	//Worms(int BPK, int TPB, GRNG &RNG, WormsParameters &wormsParameters);
	Worms();
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
	void SlowUpdate(float);
	void QuickUpdate(float);
	void CalculateThetaPhi();
	void DataDeviceToHost();
	void DataHostToDevice();
	void ResetNeighborsList();
	void ResetNeighborsList(int itime);
	void ClearAll();
	void ZeroForce();

	//.. for Debugging
	void DisplayNList();
	void DislayThetaPhi();
	void DisplayErrors();

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

	void AllocateGPUMemory_Pitched();
	void DataDeviceToHost_Pitched();
	void DataHostToDevice_Pitched();
	void ZeroGPU_Pitched();

	void DistributeWormsOnHost();
	void AdjustDistribute(float target_percent);
	void PlaceWormExplicit(int wormId, float headX, float headY, float wormAngle);
	void RandomAdheringDistribute();
	void CheckSuccess(cudaError_t err);
	void FigureBlockThreadStructure(int threadsPerBlock);

	//void KNN(int k);
	void SetNeighborsCPU();
};
// ------------------------------------------------------------------------------------------
//	PUBLIC METHODS
// ------------------------------------------------------------------------------------------
Worms::Worms() : 
r(NULL), dev_r(NULL), dev_v(NULL), dev_f(NULL), 
dev_f_old(NULL), dev_thphi(NULL) {
	DEBUG_MESSAGE("Constructor");
	//.. do necessities
	srand(time(NULL));

	//this->size_of_float = sizeof(float);
	//this->size_of_int = sizeof(int);
	//CUcontext cuContext;
	//CUdevice  cuDevice = 0;
	//cuCtxCreate(&cuContext, 0, cuDevice);
	//cuMemGetInfo(&this->memory_free, &this->memory_total);
	//cuCtxDetach(cuContext);
	//this->K = 20;
	//// Determine maximum number of query that can be treated
	//max_nb_query_traited = (memory_free * MAX_PART_OF_FREE_MEMORY_USED - size_of_float * parameters->_NPARTICLES*_D_) / (size_of_float * (_D_ + parameters->_NPARTICLES) + size_of_int * K);
	//max_nb_query_traited = min(parameters->_NPARTICLES, (max_nb_query_traited / 16) * 16);
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
	this->dev_cell = NULL;
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
	
	//.. set local state
	this->pitched_memory = usePitchedMemory;
	this->rng = gaussianRandomNumberGenerator;
	this->parameters = wormsParameters;
	this->envirn = simParameters;

	//.. init linked list
	this->dcell = 2.5f * wormsParameters->_RCUT; // so lists do not need to be set often
	this->nxcell = int(ceil(simParameters->_BOX[0] / dcell));
	this->nycell = int(ceil(simParameters->_BOX[1] / dcell));
	this->ncells = this->nxcell * this->nycell;

	//.. memory structure and host memory init
	this->FigureBlockThreadStructure(threadsPerBlock);
	this->AllocateHostMemory();
	this->ZeroHost();

	//.. choose distribute method (on host)
	if (wormsParameters->_RAD) 
		this->RandomAdheringDistribute();
	else{
		this->DistributeWormsOnHost();
		this->AdjustDistribute(0.5f);
	}

	//.. choose memory allocation methods
	if (usePitchedMemory){
		this->AllocateGPUMemory_Pitched();
		this->ZeroGPU_Pitched();
	}
	else {
		this->AllocateGPUMemory();
		this->ZeroGPU();
	}

	//.. transfer to GPU and prep for run
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
}
//-------------------------------------------------------------------------------------------
void Worms::InternalForces(){
	DEBUG_MESSAGE("InternalForces");

	float noise = sqrtf(2.0f * envirn->_NSTEPS_INNER * parameters->_GAMMA * parameters->_KBT / envirn->_DT);
	int N = _D_ * this->parameters->_NPARTICLES;
	float * rng_ptr = this->rng->Get(N);
	InterForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift,
		this->dev_v, this->vshift,
		this->dev_r, this->rshift,
		rng_ptr, noise
	);
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

	dim3 gridStruct(this->Blocks_Per_Kernel, _D_);
	dim3 blockStruct(this->Threads_Per_Block);
	int N = _D_ * this->parameters->_NPARTICLES;
	float noise = sqrtf(2.0f * parameters->_GAMMA * parameters->_KBT / envirn->_DT);
	NoiseKernel <<< gridStruct, blockStruct >>>
	(
		this->dev_f, this->fshift,
		this->rng->Generate(N), noise
	);
}
//-------------------------------------------------------------------------------------------
void Worms::LJForces(){
	DEBUG_MESSAGE("LJForces");
	if (this->parameters->_NOINT) return; // stop if not needed

	LennardJonesNListKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift, 
		this->dev_r, this->rshift, 
		this->dev_nlist, this->nlshift
	);

	//const int nblocks = this->Blocks_Per_Kernel;
	//const int nthreads = this->Threads_Per_Block;
	//const int nmax = this->parameters->_NMAX;
	//const int nparts = this->parameters->_NPARTICLES;

	//dim3 threadStruct( nmax/8, 8); // nmax should be 2^n value
	//dim3 numBlocks((int)sqrt(nparts) + 1, (int)(nparts) + 1);
	//FastLennardJonesNListKernel <<< numBlocks, threadStruct, sizeof(float)*_D_*nmax >>>
	//(
	//	this->dev_f, this->fshift,
	//	this->dev_r, this->rshift,
	//	this->dev_nlist, this->nlshift
	//);
}
//-------------------------------------------------------------------------------------------
void Worms::AutoDriveForces(int itime, int istart = 0){
	DEBUG_MESSAGE("AutoDriveForces");
	if (itime < istart) return;
	//this->CalculateThetaPhi();

	DriveForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift, 
		/*this->dev_thphi, this->tpshift*/
		this->dev_r, this->rshift
	);
}
//-------------------------------------------------------------------------------------------
void Worms::LandscapeForces(){
	DEBUG_MESSAGE("LandscapeForces");

	WormsLandscapeKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift,
		this->dev_r, this->rshift
	);
}
//-------------------------------------------------------------------------------------------
void Worms::SlowUpdate(float Amp){
	DEBUG_MESSAGE("SlowUpdate");

	UpdateSystemKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift, 
		this->dev_f_old, this->foshift, 
		this->dev_v, this->vshift, 
		this->dev_r, this->rshift,
		this->dev_cell, this->cshift,
		this->envirn->_DT, Amp
	);
	/*
	dim3 gridStruct(this->Blocks_Per_Kernel, _D_);
	dim3 blockStruct(this->Threads_Per_Block);
	FastUpdateKernel <<< gridStruct, blockStruct >>>
	(
		this->dev_f, this->fshift,
		this->dev_f_old, this->foshift,
		this->dev_v, this->vshift,
		this->dev_r, this->rshift,
		this->dev_cell, this->cshift,
		this->envirn->_DT
	);*/
}
//-------------------------------------------------------------------------------------------
void Worms::QuickUpdate(float Amp){
	DEBUG_MESSAGE("QuickUpdate");
	const float increaseRatio = (float)this->envirn->_NSTEPS_INNER;
	UpdateSystemKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fshift,
		this->dev_f_old, this->foshift,
		this->dev_v, this->vshift,
		this->dev_r, this->rshift,
		this->dev_cell, this->cshift,
		this->envirn->_DT / increaseRatio,
		Amp
	);

	/*
	const float increaseRatio = (float)this->envirn->_NSTEPS_INNER;
	dim3 gridStruct(this->Blocks_Per_Kernel, _D_);
	dim3 blockStruct(this->Threads_Per_Block);
	FastUpdateKernel <<< gridStruct, blockStruct >>>
	(
		this->dev_f, this->fshift,
		this->dev_f_old, this->foshift,
		this->dev_v, this->vshift,
		this->dev_r, this->rshift,
		this->dev_cell, this->cshift,
		this->envirn->_DT / increaseRatio
	);*/
}
//-------------------------------------------------------------------------------------------
void Worms::CalculateThetaPhi(){
	DEBUG_MESSAGE("CalculateThetaPhi");

	CalculateThetaKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_r, this->rshift, 
		this->dev_thphi, this->tpshift
	);

	FinishCalculateThetaKernel <<< (this->parameters->_NWORMS / this->Threads_Per_Block) + 1, this->Threads_Per_Block >>>
	(
		this->dev_thphi, this->tpshift
	);
}
//-------------------------------------------------------------------------------------------
void Worms::DataDeviceToHost(){
	DEBUG_MESSAGE("DataDeviceToHost");

	if (this->pitched_memory) 
		this->DataDeviceToHost_Pitched();
	else {
		CheckSuccess(cudaMemcpy(this->r, 
			this->dev_r, 
			_D_ * this->nparticles_float_alloc,
			cudaMemcpyDeviceToHost));
		ErrorHandler(cudaGetLastError());
	}
}
//-------------------------------------------------------------------------------------------
void Worms::DataHostToDevice(){
	DEBUG_MESSAGE("DataHostToDevice");

	if (this->pitched_memory)
		this->DataHostToDevice_Pitched();
	else {
		CheckSuccess(cudaMemcpy(this->dev_r, this->r, _D_ * this->nparticles_float_alloc, cudaMemcpyHostToDevice));
		ErrorHandler(cudaGetLastError());
	}
}
//-------------------------------------------------------------------------------------------
//.. sets neighbors list between worm-worm and worm-fluid
void Worms::ResetNeighborsList(){
	DEBUG_MESSAGE("ResetNeighborsList");
	if (this->parameters->_NOINT) return; // stop if not needed

	//this->SetNeighborsCPU();

	//.. reset list memory to -1
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
	//ErrorHandler(cudaDeviceSynchronize());

	//.. assign neighbors
	SetNeighborList_N2Kernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_r, this->rshift,
		this->dev_nlist, this->nlshift,
		this->dev_cell, this->cshift
	);
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

	//ErrorHandler(cudaDeviceSynchronize());
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
	};

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
}
// ------------------------------------------------------------------------------------------
//	PRIVATE METHODS
// ------------------------------------------------------------------------------------------
void Worms::AllocateHostMemory(){
	DEBUG_MESSAGE("AllocateHostMemory");
	this->r = new float[_D_*this->parameters->_NPARTICLES];
	this->c = new char[this->parameters->_NPARTICLES];
	//this->heads = new int[this->ncells];
	//this->ptz = new int[this->parameters->_NPARTICLES];
	//this->host_nlist = new int[this->parameters->_NPARTICLES*this->parameters->_NMAX];
	//this->host_nlist_cntr = new int[this->parameters->_NPARTICLES];
	//this->host_ndist = new float[this->parameters->_NPARTICLES*this->parameters->_NMAX*_D_];
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
	CheckSuccess(cudaMalloc((void**)&this->dev_ndist, this->parameters->_NMAX * this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_thphi, 2 * this->nparticles_float_alloc));
	//CheckSuccess(cudaMalloc((void**)&this->dev_cell, _D_ * this->nparticles_int_alloc));

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

	CheckSuccess(cudaMallocPitch(&this->dev_cell,
								&this->cpitch,
								this->nparticles_int_alloc,
								_D_));

	CheckSuccess(cudaMallocPitch(&this->dev_nlist,
								&this->nlpitch,
								this->nparticles_int_alloc,
								this->heightNMAX));

	CheckSuccess(cudaMallocPitch(&this->dev_ndist,
								&this->ndpitch,
								this->nparticles_float_alloc,
								this->heightNMAX*_D_)); //dist for _D_ dimensions

	CheckSuccess(cudaMallocPitch(&this->dev_thphi,
								&this->tppitch,
								widthN,
								this->height2));

	/*CheckSuccess(cudaMallocPitch(&this->dist_dev,
								&this->dist_pitch,
								this->max_nb_query_traited,
								this->parameters->_NPARTICLES));*/

	//CheckSuccess(cudaMalloc((void**)&this->dev_cell, _D_ * this->nparticles_int_alloc));

	//.. calculate and assign shifts
	this->fshift = this->fpitch / sizeof(float);
	this->foshift = this->fopitch / sizeof(float);
	this->vshift = this->vpitch / sizeof(float);
	this->rshift = this->rpitch / sizeof(float);
	this->tpshift = this->tppitch / sizeof(float);
	this->nlshift = this->nlpitch / sizeof(int);
	this->ndshift = this->ndpitch / sizeof(float);
	//this->cshift = this->cpitch / sizeof(int);

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
	const int nworms = this->parameters->_NWORMS;
	const int nparts = this->parameters->_NPARTICLES;
	const int np = this->parameters->_NP;
	const int xdim = this->parameters->_XDIM;
	const int ydim = this->parameters->_YDIM;
	const int zdim = this->parameters->_ZDIM;
	const float l1 = this->parameters->_L1;
	const float xbox = this->envirn->_XBOX;
	const float ybox = this->envirn->_YBOX;
	const float zbox = this->envirn->_ZBOX;
	const float spacing[3] = { // places worms in center of dimension
		xbox / float(xdim), // i.e. if zdim=1, z0=zbox/2
		ybox / float(ydim),
		0 //parameters->_RCUT
	};
	float * r0 = new float[_D_*nworms];
	int iw = 0;
	for (int k = 0; k < zdim; k++){
		for (int i = 0; i < xdim; i++){
			for (int j = 0; j < ydim; j++){
				const float idx[3] = { (i), (j), (k) }; // always 3d
				for_D_ r0[iw + d*nworms] = 0.001f + idx[d] * spacing[d];
				iw++;
			}
		}
	}

	// Distribute bodies
	const float s[3] = { l1, 0, 0 }; // always 3d: slope of laying chains from head
	for (int i = 0; i < nparts; i++){
		const int w = i / np;
		const int p = i % np;
		float rn[_D_], _r[_D_];
		for_D_ rn[d] = 0.25f*float(rand()) / float(RAND_MAX);
		for_D_ _r[d] = r0[w + d*nworms] + p * s[d] + rn[d];
		MovementBC(_r[0], xbox);
		MovementBC(_r[1], ybox);
		for_D_ r[i + d*nparts] = _r[d];
	}

	delete[] r0;
}
//-------------------------------------------------------------------------------------------
void Worms::AdjustDistribute(float target_percent){

	//.. grab parameters
	const int np = this->parameters->_NP;
	const int nworms = this->parameters->_NWORMS;
	const int N = np * nworms;

	//.. flip orientation for target_percent of worms
	float * save = new float[_D_ * np]; 

	for (int w = 0; w < nworms; w++){
		float random = float(rand()) / float(RAND_MAX);
		if (random <= target_percent){
			//.. store position of particles to be flipped
			for (int i = 0; i < np; i++){
				int id = w*np + i;
				for_D_ save[i + d*np] = this->r[id + d*N];
			}

			//.. replace reversed particles
			for (int j = 0; j < np; j++){
				int id = w*np + j;
				for_D_ this->r[id + d*N] = save[(np - 1 - j) + d*np];
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
void Worms::RandomAdheringDistribute(){ // 2D only

	//.. grab parameters
	printf("\n\nPlacing %i using random adhersion.", this->parameters->_NWORMS);
	const int nworms = this->parameters->_NWORMS;
	const int nparts = this->parameters->_NPARTICLES;
	const int np = this->parameters->_NP;
	const float l1 = this->parameters->_L1;
	const float xbox = this->envirn->_XBOX;
	const float ybox = this->envirn->_YBOX;
	const float zbox = this->envirn->_ZBOX;
	float * box = this->envirn->_BOX;
	const float r2min = this->parameters->_R2MIN;

	float r0[_D_], theta, phi, u[3];
	for (int w = 0; w < nworms; w++){

		float * worm = new float[_D_*np];
		for_D_ r0[d] = this->envirn->_BOX[d] * (float(rand()) / float(RAND_MAX)); // random pos in box
		for_D_ worm[0 + d*np] = r0[d]; // assing to current worm

		//.. pick random 2pi angle for theta and pi for phi for 2nd particle
		theta = 2.0f * M_PI * (float(rand()) / float(RAND_MAX));
		//phi = M_PI * (float(rand()) / float(RAND_MAX)); // uncomment for 3D
		// const float sinph = sinf(phi), cosph = cosf(phi);
		const float sinph = 1.0f, cosph = 0.0f; // for 2D debugging

		//.. connection vector
		u[0] = l1 * cosf(theta)*sinph;
		u[1] = l1 * sinf(theta)*sinph;
		u[2] = l1 * cosph;
		
		//.. place second particle dist l1 from 1st
		for_D_ worm[1 + d*np] = r0[d] + u[d]; // place second particle
		for_D_ r0[d] = worm[1 + d*np]; // new starting point to 2nd particles

		//.. loop through rest of worm
		const float maxAngle = M_PI / 10;
		for (int p = 2; p < np; p++){

			//.. pick angle with [-maxTheta, maxTheta]
			theta = maxAngle * (2 * (float(rand()) / float(RAND_MAX)) - 1);
			//phi = M_PI_2 + maxAngle * (2 * (float(rand()) / float(RAND_MAX)) - 1); // uncomment for 3D
			// const float sinph = sinf(phi), cosph = cosf(phi);
			const float sinph = 1.0f, cosph = 0.0f; // for 2D debugging

			//.. connection vector
			u[0] = l1 * cosf(theta)*sinph;
			u[1] = l1 * sinf(theta)*sinph;
			u[2] = l1 * cosph;

			for_D_ worm[p + d*np] = r0[d] + u[d];
			for_D_ r0[d] = worm[p + d*np]; // set next starting point to this worm
		}

		//.. check if any other worm has over lapping particles (not for first worm)
		bool okay = true;
		for (int w2 = 0; w2 < w; w++){ //.. loop through other placed worms
			for (int p1 = 0; p1 < np; p1++){ // particle in first worm
				for (int p2 = 0; p2 < np; p2++){ // particle in 2nd worm
					float dr[_D_], rr = 0.0f;
					const int id2 = w2*np + p2;
					for_D_ dr[d] = this->r[id2 + d*nparts] - worm[p1 + d*np];
					for_D_ PBC(dr[d], box[d]);
					for_D_ rr += dr[d] * dr[d];
					if (rr < r2min){ // flag too close
						okay = false;
					}
				}
			}
		}

		if (okay) { // set to this->r[]
			for (int p = 0; p < np; p++){
				int id = w*np + p;
				for_D_ this->r[id + d*nparts] = worm[p + d*np];
				printf("\n%i worms placed", w);
			}
		}

		delete[] worm;
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
	CheckSuccess(cudaMemset((void**)this->dev_ndist, 0, _D_*this->parameters->_NMAX * this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_cell, -1, _D_ * this->nparticles_int_alloc));
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

	CheckSuccess(cudaMemset2D(this->dev_ndist,
		this->nlpitch,
		0,
		this->nparticles_float_alloc,
		this->heightNMAX*_D_));

	CheckSuccess(cudaMemset2D(this->dev_cell,
		this->cpitch,
		-1,
		this->nparticles_int_alloc,
		_D_));

	//CheckSuccess(cudaMemset((void**)this->dev_cell, -1, _D_ * this->nparticles_int_alloc));
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
void Worms::SetNeighborsCPU(){ // always just xy domains NOT WORKING

	const int N = this->parameters->_NPARTICLES;
	const int nmax = this->parameters->_NMAX;
	int * nlist = this->host_nlist;
	int * nlist_cntr = this->host_nlist_cntr;
	float * ndist = this->host_ndist;
	const int nd_shift = N*nmax;
	const float hx = this->envirn->_BOX[0]; 
	const float hxo2 = hx / 2.0f;
	const float hy = this->envirn->_BOX[1];
	const float hyo2 = hy / 2.0f;
	const float rcut = this->parameters->_RCUT;

	//.. get data
	this->DataDeviceToHost();

	//.. reset linked list
	for (int i = 0; i < ncells; i++){
		heads[i] = -1;
	}

	//.. and neighbors lists
	for (int i = 0; i < N; i++){
		nlist_cntr[i] = 0;
		for (int n = 0; n < nmax; n++){
			nlist[i + n*N] = -1;
			for_D_ ndist[i + n*N + d*nd_shift] = 0.0f;
		}
	}

	//.. set linkage
	int icell, jcell, scell;
	float x, y, z;
	for (int i = 0; i < N; i++){
		x = r[i + N * 0];
		y = r[i + N * 1];
		//printf("{ %f, %f }\t", x, y);
		icell = int(floor(x / dcell));
		jcell = int(floor(y / dcell));
		scell = icell + jcell*nxcell;
		ptz[i] = heads[scell]; // point particle i to old head
		heads[scell] = i; // set new head as particle i
	}

	//.. construct neighbors list
	const int dic[5] = { 0,  1, 1, 1, 0 };
	const int djc[5] = { 0, -1, 0, 1, 1 };
	int inc, jnc, snc, sc, ii, jj, icntr, jcntr;
	float dx, dy, dz, rr;
	for (int ic = 0; ic < nxcell; ic++){ // loop cells i
		for (int jc = 0; jc < nycell; jc++){ //loop cells j
			sc = ic + jc*nxcell;
			if (heads[sc] == -1) continue; // stop if emtpy cell
			//printf("\n\nBOX %d ( %d, %d ) -----------------------------------------", sc, ic, jc);
			for (int d = 0; d < 5; d++){ // look in 'neighbor cells
				inc = ic + dic[d] % nxcell;
				jnc = jc + djc[d] % nycell;
				if (inc < 0) inc = (nxcell - 1);
				if (jnc < 0) jnc = (nycell - 1);
				snc = inc + jnc*nxcell;
				//printf("\n\n   BOX %d ( %d, %d ) <- d[ %d, %d ] ", snc, inc, jnc, dic[d], djc[d]);
				if (heads[snc] == -1) continue; // stop if this cell is empty
				ii = heads[sc];
				while (ii >= 0){
					//printf("\n\n     PARTICLES (%d):", ii);
					if (nlist_cntr[ii] >= nmax){
						ii = ptz[ii];
						continue;
					}
					jj = heads[snc];
					while (jj >= 0){
						if (ii == jj){
							jj = ptz[jj];
							continue;
						}
						//printf("\t%d", jj);
						if (nlist_cntr[jj] >= nmax){
							jj = ptz[jj];
							continue;
						}
						dx = r[ii + N * 0] - r[jj + N * 0];
						dy = r[ii + N * 1] - r[jj + N * 1];
						dz = r[ii + N * 2] - r[jj + N * 2];
						if (dx > hxo2) dx -= hx;
						else if (dx < -hxo2) dx += hx;
						if (dy > hyo2) dy -= hy;
						else if (dy < -hyo2) dy += hy;
						rr = dx*dx + dy*dy + dz*dz;
						if (rr > dcell*dcell) {
							jj = ptz[jj];
							continue;
						}
						// ii particle
						icntr = nlist_cntr[ii]; // previous neighbor count
						nlist[ii + N*icntr] = jj; // save index
						ndist[ii + N*icntr + nd_shift * 0] = -dx; // save disp comps
						ndist[ii + N*icntr + nd_shift * 1] = -dy;
						ndist[ii + N*icntr + nd_shift * 2] = -dz;
						nlist_cntr[ii]++; // add to neighbor count
						// jj particle
						jcntr = nlist_cntr[jj];
						nlist[jj + N*jcntr] = ii;
						ndist[jj + N*jcntr + nd_shift * 0] = dx;
						ndist[jj + N*jcntr + nd_shift * 1] = dy;
						ndist[jj + N*jcntr + nd_shift * 2] = dz;
						nlist_cntr[jj]++;

						jj = ptz[jj];
					}
					ii = ptz[ii];
				}
			}
		}
	}

	/*for (int p = 0; p < N; p++){
		printf("\n\nP %i @ { %f, %f, %f }", 
			p, 
			r[p + 0 * N], 
			r[p + 1 * N], 
			r[p + 2 * N]);
		for (int n = 0; n < nlist_cntr[p]; n++){
			printf("\n\n  N %d (%d) dr = { %f, %f, %f }", 
				n, 
				nlist[p + N*n], 
				ndist[p + N*n + nd_shift * 0], 
				ndist[p + N*n + nd_shift * 1], 
				ndist[p + N*n + nd_shift * 2]);
		}
	}*/

	//.. copy data to gpu
	CheckSuccess(cudaMemcpy2D(this->dev_nlist,
		this->nlpitch,
		this->host_nlist,
		this->nparticles_int_alloc,
		this->nparticles_int_alloc,
		nmax,
	cudaMemcpyHostToDevice));

	CheckSuccess(cudaMemcpy2D(this->dev_ndist,
		this->ndpitch,
		this->host_ndist,
		this->nparticles_float_alloc,
		this->nparticles_float_alloc,
		nmax*_D_,
		cudaMemcpyHostToDevice));
}
//-------------------------------------------------------------------------------------------
#endif