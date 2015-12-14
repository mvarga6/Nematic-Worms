
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

class ForceXchanger;
// -----------------------------------------------------------------------------------------
//	This class defines a set of active flexible worms.
// -----------------------------------------------------------------------------------------
class Worms {	

	//.. on gpu device
	float * dev_X;
	float * dev_Y;
	float * dev_Z;
	float * dev_Vx;
	float * dev_Vy;
	float * dev_Vz;
	float * dev_Fx;
	float * dev_Fy;
	float * dev_Fz;
	float * dev_Fx_old;
	float * dev_Fy_old;
	float * dev_Fz_old;
	float * dev_theta;
	float * dev_phi;

	//.. neighbors list within worms
	int * dev_nlist;

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
	float * X;
	float * Y;
	float * Z;
	//float * Vx;
	//float * Vy;
	//float * Vz;

	//.. construct with block-thread structure
	//Worms(int BPK, int TPB, GRNG &RNG, WormsParameters &wormsParameters);
	Worms();
	~Worms();

	//.. user access to running simulation
	void Init(GRNG *, WormsParameters *, SimulationParameters *, int);
	void CustomInit(float *headX, float *headY, float *wormAngle);
	void InternalForces();
	void LJForces();
	void AutoDriveForces();
	void LandscapeForces();
	void Update();
	void CalculateTheta();
	void DataDeviceToHost();
	void DataHostToDevice();
	void ResetNeighborsList();
	void ResetNeighborsList(int itime);
	void ClearAll();

	//.. for Debugging
	void DisplayNList();
	void DislayTheta();
	void DisplayErrors();

	//.. so force exchanger can do what i needs
	friend class ForceXchanger;

private:
	//.. methods for internal use
	void AllocateHostMemory();
	void FreeHostMemory();
	void AllocateGPUMemory();
	void FreeGPUMemory();
	void DistributeWormsOnHost();
	void PlaceWormExplicit(int wormId, float headX, float headY, float wormAngle);
	void AdjustDistribute(float target_percent);
	void ZeroHost();
	void ZeroGPU();
	void CheckSuccess(cudaError_t err);
	void FigureBlockThreadStructure(int threadsPerBlock);
};
// ------------------------------------------------------------------------------------------
//	PUBLIC METHODS
// ------------------------------------------------------------------------------------------
//Worms::Worms(int BPK, int TPB, GRNG &RNG, WormsParameters &wormsParameters) :
Worms::Worms() :
X(NULL), Y(NULL), Z(NULL),/*Vx(NULL), Vy(NULL),*/ dev_theta(NULL),
rng(NULL), parameters(NULL), envirn(NULL),
dev_X(NULL), dev_Y(NULL), dev_Z(NULL),
dev_Vx(NULL), dev_Vy(NULL), dev_Vz(NULL),
dev_Fx(NULL), dev_Fy(NULL), dev_Fz(NULL),
dev_Fx_old(NULL), dev_Fy_old(NULL), dev_Fz_old(NULL){

	//.. do necessities
	srand(time(NULL));
	/*Threads_Per_Block = TPB;
	Blocks_Per_Kernel = BPK;
	this->nparticles_int_alloc = wormsParameters._NPARTICLES * sizeof(int);
	this->nparticles_float_alloc = wormsParameters._NPARTICLES * sizeof(float);*/
}

Worms::~Worms(){
	this->ClearAll();
}

void Worms::ClearAll(){
	this->FreeGPUMemory();
	this->FreeHostMemory();
	this->X = this->Y = this->Z = NULL;
	//this->Vx = this->Vy = NULL;
	this->dev_X = this->dev_Y = this->dev_Z = NULL;
	this->dev_Vx = this->dev_Vy = this->dev_Vz = NULL;
	this->dev_Fx = this->dev_Fy = this->dev_Fz = NULL;
	this->dev_Fx_old = this->dev_Fy_old = this->dev_Fz_old = NULL;
	this->dev_theta = NULL;
	this->rng = NULL;
	this->parameters = NULL;
	this->envirn = NULL;
}

/*
*	Initializes the worms to run using a /p gaussianRandomNumberGenerator
*	with the parameters specified by /p wormsParameters and utilizeding 
*	an envirnment specified by /p simParameters.  If /p threadsPerBlock
*	is specified, the it will be used, otherwise default is 512.  The init
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
				int threadsPerBlock = 512){
	this->rng = gaussianRandomNumberGenerator;
	this->parameters = wormsParameters;
	this->envirn = simParameters;
	this->FigureBlockThreadStructure(threadsPerBlock);
	this->AllocateHostMemory();
	this->ZeroHost();
	this->AllocateGPUMemory();
	this->ZeroGPU();
	this->DistributeWormsOnHost();
	this->AdjustDistribute(0.5f);
	this->DataHostToDevice();
	this->CalculateTheta();
	this->ResetNeighborsList();
}

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

void Worms::InternalForces(){
	InterForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_Fx, this->dev_Fy, this->dev_Fz,
		this->dev_Vx, this->dev_Vy, this->dev_Vz, 
		this->dev_X, this->dev_Y, this->dev_Z, 
		this->rng->Get(3 * this->parameters->_NPARTICLES)
	);
	ErrorHandler(cudaGetLastError());
}

void Worms::LJForces(){
	LennardJonesNListKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_Fx, this->dev_Fy, this->dev_Fz, 
		this->dev_X, this->dev_Y, this->dev_Z, 
		this->dev_nlist
	);
	ErrorHandler(cudaGetLastError());
}

void Worms::AutoDriveForces(){
	this->CalculateTheta();
	DriveForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_Fx, this->dev_Fy, this->dev_Fz, 
		this->dev_theta, this->dev_phi
	);
}

void Worms::LandscapeForces(){
	WormsLandscapeKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_Fz, this->dev_Z
	);
}

void Worms::Update(){
	UpdateSystemKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_Fx, this->dev_Fy, this->dev_Fz, 
		this->dev_Fx_old, this->dev_Fy_old, this->dev_Fz_old, 
		this->dev_Vx, this->dev_Vy, this->dev_Vz, 
		this->dev_X, this->dev_Y, this->dev_Z
	);
	ErrorHandler(cudaGetLastError());
}

void Worms::CalculateTheta(){
	CalculateThetaKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_X, this->dev_Y, this->dev_Z, this->dev_theta, this->dev_phi
	);
	FinishCalculateThetaKernel <<< (this->parameters->_NWORMS / this->Threads_Per_Block) + 1, this->Threads_Per_Block >>>
	(
		this->dev_theta, this->dev_phi
	);
	ErrorHandler(cudaGetLastError());
}

void Worms::DataDeviceToHost(){
	CheckSuccess(cudaMemcpy(this->X, this->dev_X, this->nparticles_float_alloc, cudaMemcpyDeviceToHost));
	CheckSuccess(cudaMemcpy(this->Y, this->dev_Y, this->nparticles_float_alloc, cudaMemcpyDeviceToHost));
	CheckSuccess(cudaMemcpy(this->Z, this->dev_Z, this->nparticles_float_alloc, cudaMemcpyDeviceToHost));
}

void Worms::DataHostToDevice(){
	CheckSuccess(cudaMemcpy(this->dev_X, this->X, this->nparticles_float_alloc, cudaMemcpyHostToDevice));
	CheckSuccess(cudaMemcpy(this->dev_Y, this->Y, this->nparticles_float_alloc, cudaMemcpyHostToDevice));
	CheckSuccess(cudaMemcpy(this->dev_Z, this->Z, this->nparticles_float_alloc, cudaMemcpyHostToDevice));
	//CheckSuccess(cudaMemcpy(this->dev_Vx, this->Vx, this->nparticles_float_alloc, cudaMemcpyHostToDevice));
	//CheckSuccess(cudaMemcpy(this->dev_Vy, this->Vy, this->nparticles_float_alloc, cudaMemcpyHostToDevice));
}

//.. sets neighbors list between worm-worm and worm-fluid
void Worms::ResetNeighborsList(){
	CheckSuccess(cudaMemset((void**)this->dev_nlist, -1, this->parameters->_NMAX * this->nparticles_int_alloc));
	SetNeighborList_N2Kernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_X, this->dev_Y, 
		this->dev_Z, this->dev_nlist
	);
	ErrorHandler(cudaGetLastError());
}

void Worms::ResetNeighborsList(int itime){
	if (itime % parameters->_LISTSETGAP == 0)
		this->ResetNeighborsList();
}

void Worms::DislayTheta(){
	float * theta = new float[this->parameters->_NPARTICLES];
	cudaMemcpy(theta, this->dev_theta, this->nparticles_float_alloc, cudaMemcpyDeviceToHost);
	for (int i = 0; i < this->parameters->_NPARTICLES; i++){
		if(isnan(theta[i])) std::cout << i << " = NaN" << std::endl;
	}
	delete[] theta;
}

void Worms::DisplayNList(){
	int * nlist = new int[this->parameters->_NPARTICLES * this->parameters->_NMAX];
	cudaMemcpy(nlist, this->dev_nlist, this->parameters->_NMAX * this->nparticles_int_alloc, cudaMemcpyDeviceToHost);
	for (int i = 0; i < parameters->_NPARTICLES; i++){
		printf("\nParticle %i\n", i);
		for (int n = 0; n < parameters->_NMAX; n++){
			printf("\t%i == %i", n, nlist[i*parameters->_NMAX + n]);
		}
	}
	delete[] nlist;
}

void Worms::DisplayErrors(){
	if (this->errorState.size() > 0){
		printf("\nWORM ERRORS:\n");
		for (int i = 0; i < this->errorState.size(); i++)
			printf("%i -- %s\n", i, cudaGetErrorString(this->errorState[i]));
		this->errorState.empty();
	}
	this->DislayTheta();
}
// -------------------------------------------------------------------------------------------
//	PRIVATE METHODS
// -------------------------------------------------------------------------------------------
void Worms::AllocateHostMemory(){
	this->X = new float[this->parameters->_NPARTICLES];
	this->Y = new float[this->parameters->_NPARTICLES];
	this->Z = new float[this->parameters->_NPARTICLES];
	//this->Vx = new float[this->parameters->_NPARTICLES];
	//this->Vy = new float[this->parameters->_NPARTICLES];
}

void Worms::FreeHostMemory(){
	delete[] this->X, this->Y, this->Z;// , this->Vx, this->Vy;
}

void Worms::AllocateGPUMemory(){
	CheckSuccess(cudaMalloc((void**)&(this->dev_X), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Y), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Z), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Vx), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Vy), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Vz), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Fx), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Fy), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Fz), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Fx_old), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Fy_old), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_Fz_old), this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_theta, this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&this->dev_phi, this->nparticles_float_alloc));
	CheckSuccess(cudaMalloc((void**)&(this->dev_nlist), this->parameters->_NMAX * this->nparticles_int_alloc));
}

void Worms::FreeGPUMemory(){
	cudaDeviceReset();
	/*CheckSuccess(cudaFree(this->dev_X));
	CheckSuccess(cudaFree(this->dev_Y));
	CheckSuccess(cudaFree(this->dev_Vx));
	CheckSuccess(cudaFree(this->dev_Vy));
	CheckSuccess(cudaFree(this->dev_Fx));
	CheckSuccess(cudaFree(this->dev_Fy));
	CheckSuccess(cudaFree(this->dev_Fx_old));
	CheckSuccess(cudaFree(this->dev_Fy_old));
	CheckSuccess(cudaFree(this->dev_nlist));*/
}

void Worms::DistributeWormsOnHost(){

	//.. grab parameters
	int nworms = this->parameters->_NWORMS;
	int nparts = this->parameters->_NPARTICLES;
	int np = this->parameters->_NP;
	int xdim = this->parameters->_XDIM;
	int ydim = this->parameters->_YDIM;
	float l1 = this->parameters->_L1;
	float xbox = this->envirn->_XBOX;
	float ybox = this->envirn->_YBOX;

	float *x0 = new float[nworms];
	float *y0 = new float[nworms];
	float *z0 = new float[nworms];
	int iw = 0;
	for (int i = 0; i < xdim; i++)
	{
		for (int j = 0; j < ydim; j++)
		{
			x0[iw] = 0.001 + float(i)*xbox / float(xdim);
			y0[iw] = 0.001 + float(j)*ybox / float(ydim);
			z0[iw] = 0.0f;
			iw++;
		}
	}

	// Distribute bodies
	for (int i = 0; i < nparts; i++)
	{
		float noise = 0.1f*float(rand()) / float(RAND_MAX);
		int w = i / np;
		int p = i % np;
		float x = x0[w] + p * l1;
		float y = y0[w] + noise;
		MovementBC(x, xbox);
		MovementBC(y, ybox);
		this->X[i] = x;
		this->Y[i] = y;
		this->Z[i] = 0.0f;
	}

	delete[] x0;
	delete[] y0;
	delete[] z0;
}

void Worms::AdjustDistribute(float target_percent){

	//.. grab parameters
	int np = this->parameters->_NP;
	int nworms = this->parameters->_NWORMS;

	//.. flip orientation for target_percent of worms
	float * savex = new float[np]; 
	float * savey = new float[np];

	for (int w = 0; w < nworms; w++)
	{
		float random = float(rand()) / float(RAND_MAX);
		if (random <= target_percent)
		{
			//.. store position of particles to be flipped
			for (int i = 0; i < np; i++)
			{
				savex[i] = this->X[w*np + i];
				savey[i] = this->Y[w*np + i];
			}

			//.. replace reversed particles
			for (int j = 0; j < np; j++)
			{
				this->X[w*np + j] = savex[np - 1 - j];
				this->Y[w*np + j] = savey[np - 1 - j];
			}
		}
	}
}

void Worms::PlaceWormExplicit(int wormId, float headX, float headY, float wormAngle){
	
	//.. grab parameters
	float l1 = this->parameters->_L1;
	int np = this->parameters->_NP;
	int nworms = this->parameters->_NWORMS;

	if (wormId >= nworms || wormId < 0) return;

	//.. get index of head particle
	int head_id = wormId * np;

	//.. calculate displacement based on angle
	const float dx = l1 * cosf(wormAngle);
	const float dy = l1 * sinf(wormAngle);

	//.. place head particle first
	this->X[head_id] = headX; 
	this->Y[head_id] = headY;

	//.. place the rest at the previous pos plus dx,dy
	for (int id = head_id + 1; id < (head_id + np); id++){
		this->X[id] = this->X[id - 1] + dx;
		this->Y[id] = this->Y[id - 1] + dy;
		MovementBC(this->X[id], this->envirn->_XBOX);
		MovementBC(this->Y[id], this->envirn->_YBOX);
	}
}

void Worms::ZeroHost(){
	for (int i = 0; i < this->parameters->_NPARTICLES; i++)
		this->X[i] = this->Y[i] = this->Z[i] = 0.0f;// = this->Vx[i] = this->Vy[i] = 0.0f;
}

void Worms::ZeroGPU(){
	CheckSuccess(cudaMemset((void**)this->dev_X, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Y, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Z, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Vx, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Vy, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Vz, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Fx, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Fy, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Fz, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Fx_old, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Fy_old, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_Fz_old, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_theta, 0, this->nparticles_float_alloc));
	CheckSuccess(cudaMemset((void**)this->dev_phi, 0, this->nparticles_float_alloc));
}

void Worms::CheckSuccess(cudaError_t err){
	if (err != cudaSuccess) this->errorState.push_back(err);
}

void Worms::FigureBlockThreadStructure(int tpb){
	if (tpb >= 0) this->Threads_Per_Block = tpb;
	else this->Threads_Per_Block = 256; // good default value
	this->Blocks_Per_Kernel = this->parameters->_NPARTICLES / this->Threads_Per_Block + 1; // add one to guarentee enough
	this->nparticles_float_alloc = this->parameters->_NPARTICLES * sizeof(float);
	this->nparticles_int_alloc = this->parameters->_NPARTICLES * sizeof(int);
	printf("\n\tWorms:\n\tTotalThreads = %i\n\tBlocksPerKernel = %i\n\tThreadsPerBlock = %i\n", this->parameters->_NPARTICLES, this->Blocks_Per_Kernel, this->Threads_Per_Block);
}

#endif