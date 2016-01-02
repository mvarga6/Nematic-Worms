
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
	//float * dev_X;
	//float * dev_Y;
	//float * dev_Z;
	//float * dev_Vx;
	//float * dev_Vy;
	//float * dev_Vz;
	//float * dev_Fx;
	//float * dev_Fy;
	//float * dev_Fz;
	//float * dev_Fx_old;
	//float * dev_Fy_old;
	//float * dev_Fz_old;
	//float * dev_theta;
	//float * dev_phi;

	//.. new data structure
	float * dev_r;
	float * dev_v;
	float * dev_f;
	float * dev_f_old;
	float * dev_thphi;

	//.. neighbors list within worms
	int * dev_nlist;

	//.. pitchs for memory
	size_t rpitch;
	size_t vpitch;
	size_t fpitch;
	size_t fopitch;
	size_t tppitch;
	size_t nlpitch;

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

public:
	//.. on host (so print on host is easier)
	//float * X;
	//float * Y;
	//float * Z;
	float * r;
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
	void NoiseForces();
	void LJForces();
	void AutoDriveForces();
	void LandscapeForces();
	void Update();
	void CalculateThetaPhi();
	void DataDeviceToHost();
	void DataHostToDevice();
	void ResetNeighborsList();
	void ResetNeighborsList(int itime);
	void ClearAll();

	//.. for Debugging
	void DisplayNList();
	void DislayThetaPhi();
	void DisplayErrors();
	void ZeroForce();
	void AddConstantForce(int dim, float force);

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
r(NULL), dev_r(NULL), dev_v(NULL), dev_f(NULL), 
dev_f_old(NULL), dev_thphi(NULL)
/*X(NULL), Y(NULL), Z(NULL),Vx(NULL), Vy(NULL), dev_theta(NULL),
rng(NULL), parameters(NULL), envirn(NULL),
dev_X(NULL), dev_Y(NULL), dev_Z(NULL),
dev_Vx(NULL), dev_Vy(NULL), dev_Vz(NULL),
dev_Fx(NULL), dev_Fy(NULL), dev_Fz(NULL),
dev_Fx_old(NULL), dev_Fy_old(NULL), dev_Fz_old(NULL)*/{

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
	//this->X = this->Y = this->Z = NULL;
	//this->Vx = this->Vy = NULL;
	//this->dev_X = this->dev_Y = this->dev_Z = NULL;
	//this->dev_Vx = this->dev_Vy = this->dev_Vz = NULL;
	//this->dev_Fx = this->dev_Fy = this->dev_Fz = NULL;
	//this->dev_Fx_old = this->dev_Fy_old = this->dev_Fz_old = NULL;
	//this->dev_theta = NULL;
	this->r = NULL;
	this->dev_r = this->dev_v = NULL;
	this->dev_f = this->dev_f_old = NULL;
	this->dev_thphi = NULL;
	this->rng = NULL;
	this->parameters = NULL;
	this->envirn = NULL;
}

/*
*	Initializes the worms to run using a /p gaussianRandomNumberGenerator
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
				int threadsPerBlock = 256){
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
	this->CalculateThetaPhi();
	this->ResetNeighborsList();
	ErrorHandler(cudaDeviceSynchronize());
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
	ErrorHandler(cudaDeviceSynchronize());
}

void Worms::InternalForces(){
	float noise = sqrtf(2.0f * parameters->_GAMMA * parameters->_KBT / envirn->_DT);
	int N = 3 * this->parameters->_NPARTICLES;
	InterForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fpitch / sizeof(float),
		this->dev_v, this->vpitch / sizeof(float),
		this->dev_r, this->rpitch / sizeof(float),
		this->rng->GenerateAll(), noise
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

void Worms::NoiseForces(){

	dim3 gridStruct(this->Blocks_Per_Kernel, 3);
	dim3 blockStruct(this->Threads_Per_Block);
	int N = 3 * this->parameters->_NPARTICLES;
	float noise = sqrtf(2.0f * parameters->_GAMMA * parameters->_KBT / envirn->_DT);
	NoiseKernel <<< gridStruct, blockStruct >>>
	(
		this->dev_f, this->fpitch / sizeof(float),
		this->rng->Generate(N), noise
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

void Worms::LJForces(){
	LennardJonesNListKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fpitch, 
		this->dev_r, this->rpitch, 
		this->dev_nlist, this->nlpitch
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

void Worms::AutoDriveForces(){
	this->CalculateThetaPhi();
	DriveForceKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fpitch, 
		this->dev_thphi, this->tppitch
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

void Worms::LandscapeForces(){
	WormsLandscapeKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fpitch,
		this->dev_r, this->rpitch
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

void Worms::Update(){
	/*UpdateSystemKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_f, this->fpitch, 
		this->dev_f_old, this->fopitch, 
		this->dev_v, this->vpitch, 
		this->dev_r, this->rpitch
	);*/

	dim3 gridStruct(this->Blocks_Per_Kernel, 3);
	dim3 blockStruct(this->Threads_Per_Block);
	FastUpdateKernel <<< gridStruct, blockStruct >>>
	(
		this->dev_f, this->fpitch / sizeof(float),
		this->dev_f_old, this->fopitch / sizeof(float),
		this->dev_v, this->vpitch / sizeof(float),
		this->dev_r, this->rpitch / sizeof(float)
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

void Worms::CalculateThetaPhi(){
	CalculateThetaKernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_r, this->rpitch, 
		this->dev_thphi, this->tppitch
	);
	ErrorHandler(cudaDeviceSynchronize());
	FinishCalculateThetaKernel <<< (this->parameters->_NWORMS / this->Threads_Per_Block) + 1, this->Threads_Per_Block >>>
	(
		this->dev_thphi, this->tppitch
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

void Worms::DataDeviceToHost(){
	//CheckSuccess(cudaMemcpy(this->X, this->dev_X, this->nparticles_float_alloc, cudaMemcpyDeviceToHost));
	//CheckSuccess(cudaMemcpy(this->Y, this->dev_Y, this->nparticles_float_alloc, cudaMemcpyDeviceToHost));
	//CheckSuccess(cudaMemcpy(this->Z, this->dev_Z, this->nparticles_float_alloc, cudaMemcpyDeviceToHost));
	
	CheckSuccess(cudaMemcpy2D(this->r,
		this->nparticles_float_alloc,
		this->dev_r,
		this->rpitch,
		this->nparticles_float_alloc,
		this->height3,
		cudaMemcpyDeviceToHost));
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

void Worms::DataHostToDevice(){
	//CheckSuccess(cudaMemcpy(this->dev_X, this->X, this->nparticles_float_alloc, cudaMemcpyHostToDevice));
	//CheckSuccess(cudaMemcpy(this->dev_Y, this->Y, this->nparticles_float_alloc, cudaMemcpyHostToDevice));
	//CheckSuccess(cudaMemcpy(this->dev_Z, this->Z, this->nparticles_float_alloc, cudaMemcpyHostToDevice));
	//CheckSuccess(cudaMemcpy(this->dev_Vx, this->Vx, this->nparticles_float_alloc, cudaMemcpyHostToDevice));
	//CheckSuccess(cudaMemcpy(this->dev_Vy, this->Vy, this->nparticles_float_alloc, cudaMemcpyHostToDevice));

	CheckSuccess(cudaMemcpy2D(this->dev_r,
		this->rpitch,
		this->r,
		this->nparticles_float_alloc,
		this->nparticles_float_alloc,
		this->height3,
		cudaMemcpyHostToDevice));
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

//.. sets neighbors list between worm-worm and worm-fluid
void Worms::ResetNeighborsList(){
	
	CheckSuccess(cudaMemset2D(this->dev_nlist,
		this->nlpitch,
		-1, 
		this->nparticles_int_alloc,
		this->heightNMAX));
	ErrorHandler(cudaDeviceSynchronize());

	SetNeighborList_N2Kernel <<< this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
	(
		this->dev_r, this->rpitch,
		this->dev_nlist, this->nlpitch
	);
	ErrorHandler(cudaDeviceSynchronize());
	ErrorHandler(cudaGetLastError());
}

void Worms::ResetNeighborsList(int itime){
	if (itime % parameters->_LISTSETGAP == 0)
		this->ResetNeighborsList();
}

void Worms::DislayThetaPhi(){

	float * thphi = new float[2 * this->parameters->_NPARTICLES];

	CheckSuccess(cudaMemcpy2D(thphi, 
		this->nparticles_float_alloc,
		this->dev_thphi, 
		this->tppitch,
		this->nparticles_float_alloc,
		this->height2, 
		cudaMemcpyDeviceToHost));
	
	for (int i = 0; i < this->parameters->_NPARTICLES; i++){
		/*if (isnan(thphi[i]))
			std::cout << "thphi[" << i << "] = NaN" << std::endl;
		if (isnan(thphi[i + this->parameters->_NPARTICLES])) 
			std::cout << "thphi[" << i << "] = NaN" << std::endl;*/
		std::cout << "\nth = " << thphi[i] << "\tphi = " << thphi[i + this->parameters->_NPARTICLES];
	}
	delete[] thphi;
}

void Worms::DisplayNList(){

	size_t size = this->parameters->_NPARTICLES * this->parameters->_NMAX;
	int * nlist = new int[size];

	CheckSuccess(cudaMemcpy2D(nlist,
		this->nparticles_int_alloc,
		this->dev_nlist,
		this->nlpitch,
		this->nparticles_int_alloc,
		this->heightNMAX,
		cudaMemcpyDeviceToHost));

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

void Worms::DisplayErrors(){
	if (this->errorState.size() > 0){
		printf("\nWORM ERRORS:\n");
		for (int i = 0; i < this->errorState.size(); i++)
			printf("%i -- %s\n", i, cudaGetErrorString(this->errorState[i]));
		this->errorState.empty();
	}
	//this->DislayThetaPhi();
}

void Worms::ZeroForce(){
	CheckSuccess(cudaMemset2D(this->dev_f,
		this->fpitch,
		0,
		this->parameters->_NPARTICLES*sizeof(float),
		this->height3));
	ErrorHandler(cudaDeviceSynchronize());
}

void Worms::AddConstantForce(int dim, float force){
	AddForce <<<  this->Blocks_Per_Kernel, this->Threads_Per_Block >>>
		(
		this->dev_f, this->fpitch,
		dim, force
		);
	ErrorHandler(cudaDeviceSynchronize());
}

// -------------------------------------------------------------------------------------------
//	PRIVATE METHODS
// -------------------------------------------------------------------------------------------
void Worms::AllocateHostMemory(){
	//this->X = new float[this->parameters->_NPARTICLES];
	//this->Y = new float[this->parameters->_NPARTICLES];
	//this->Z = new float[this->parameters->_NPARTICLES];
	//this->Vx = new float[this->parameters->_NPARTICLES];
	//this->Vy = new float[this->parameters->_NPARTICLES];
	this->r = new float[3*this->parameters->_NPARTICLES];
}

void Worms::FreeHostMemory(){
	delete[] this->r; // , this->Vx, this->Vy;
}

void Worms::AllocateGPUMemory(){
	/*CheckSuccess(cudaMalloc((void**)&(this->dev_X), this->nparticles_float_alloc));
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
	CheckSuccess(cudaMalloc((void**)&(this->dev_nlist), this->parameters->_NMAX * this->nparticles_int_alloc));*/
	
	this->height3 = 3;
	this->height2 = 2;
	this->heightNMAX = this->parameters->_NMAX;
	size_t widthN = this->nparticles_float_alloc;

	CheckSuccess(cudaMallocPitch(&this->dev_r, 
								&this->rpitch, 
								widthN,
								this->height3));

	CheckSuccess(cudaMallocPitch(&this->dev_v,
								&this->vpitch,
								widthN,
								this->height3));

	CheckSuccess(cudaMallocPitch(&this->dev_f,
								&this->fpitch,
								widthN,
								this->height3));

	CheckSuccess(cudaMallocPitch(&this->dev_f_old,
								&this->fopitch,
								widthN,
								this->height3));

	CheckSuccess(cudaMallocPitch(&this->dev_nlist,
								&this->nlpitch,
								this->nparticles_int_alloc,
								this->heightNMAX));

	CheckSuccess(cudaMallocPitch(&this->dev_thphi,
								&this->tppitch,
								widthN,
								this->height2));

	printf("\nDevice memory pitches:\n----------------------\nf:\t%i\nf_old:\t%i\nv:\t%i\nr:\t%i\nthephi:\t%i\nnlist:\t%i\n",
		this->fpitch, 
		this->fopitch, 
		this->vpitch, 
		this->rpitch,
		this->tppitch,
		this->nlpitch);

	printf("\nDevice memory elements:\n-----------------------\nf:\t%i\nf_old:\t%i\nv:\t%i\nr:\t%i\nthephi:\t%i\nnlist:\t%i\n",
		this->fpitch / sizeof(float), 
		this->fopitch / sizeof(float), 
		this->vpitch / sizeof(float), 
		this->rpitch / sizeof(float),
		this->tppitch / sizeof(float),
		this->nlpitch / sizeof(int));

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
		float rx = 0.25f*float(rand()) / float(RAND_MAX);
		float ry = 0.25f*float(rand()) / float(RAND_MAX);
		float rz = 0.25f*float(rand()) / float(RAND_MAX);
		int w = i / np;
		int p = i % np;
		float x = x0[w] + p * l1;
		float y = y0[w];
		MovementBC(x, xbox);
		MovementBC(y, ybox);
		this->r[i] = x + rx;
		this->r[i + nparts] = y + ry;
		this->r[i + 2*nparts] = rz;
	}

	delete[] x0;
	delete[] y0;
	delete[] z0;
}

void Worms::AdjustDistribute(float target_percent){

	//.. grab parameters
	int np = this->parameters->_NP;
	int nworms = this->parameters->_NWORMS;
	int N = np * nworms;

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
				savex[i] = this->r[(w*np + i)];
				savey[i] = this->r[(w*np + i) + N];
			}

			//.. replace reversed particles
			for (int j = 0; j < np; j++)
			{
				this->r[(w*np + j)] = savex[(np - 1 - j)];
				this->r[(w*np + j) + N] = savey[(np - 1 - j)];
			}
		}
	}
}

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

void Worms::ZeroHost(){
	for (int i = 0; i < 3 *this->parameters->_NPARTICLES; i++)
		this->r[i] = 0.0f;
}

void Worms::ZeroGPU(){
	/*CheckSuccess(cudaMemset((void**)this->dev_X, 0, this->nparticles_float_alloc));
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
	CheckSuccess(cudaMemset((void**)this->dev_phi, 0, this->nparticles_float_alloc));*/

	size_t widthN = this->nparticles_float_alloc;

	CheckSuccess(cudaMemset2D(this->dev_r,
		this->rpitch,
		0,
		widthN,
		this->height3));

	CheckSuccess(cudaMemset2D(this->dev_v,
		this->vpitch,
		0,
		widthN,
		this->height3));

	CheckSuccess(cudaMemset2D(this->dev_f,
		this->fpitch,
		0,
		widthN,
		this->height3));

	CheckSuccess(cudaMemset2D(this->dev_f_old,
		this->fopitch,
		0,
		widthN,
		this->height3));

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
	printf("\nWorms:\n------\nTotalThreads = %i\nBlocksPerKernel = %i\nThreadsPerBlock = %i", 
		this->parameters->_NPARTICLES, 
		this->Blocks_Per_Kernel, 
		this->Threads_Per_Block);
	printf("\nfloat allocs = %i\nint allocs = %i\n",
		this->nparticles_float_alloc,
		this->nparticles_int_alloc);
}

#endif