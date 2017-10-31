#include "../../../include/physics/HoegerPhysics.h"


NW::HoegerPhysics::HoegerPhysics(KernelLaunchInfo * kernelLaunchInfo)
{
	this->launchInfo = kernelLaunchInfo;
}

NW::HoegerPhysics::~HoegerPhysics()
{
}

void NW::HoegerPhysics::InternalForces(real * f, int fshift, real * v, int vshift, real * r, int rshift, float * randNum, float noiseScaler)
{
	NW::HoegerPhysics::__InternalForces
		<<<launchInfo->ParticleGridStructure, launchInfo->ParticleBlockStructure >>>
		(f, fshift, v, vshift, r, rshift, randNum, noiseScaler);
}

__global__ void NW::HoegerPhysics::__InternalForces(real * f, int fshift, real * v, int vshift, real * r, int rshift, float * randNum, float noiseScaler)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES) {

		//.. particel number in worm
		int p = id % dev_Params._NP;
		int w = id / dev_Params._NP;

		//.. local memory
		float rid[_D_], fid[_D_], vid[_D_];

		//.. init local as needed
		for_D_{
			rid[d] = r[id + d*rshift];
		fid[d] = 0.0f;
		vid[d] = v[id + d*vshift];
		}

			//.. 1st neighbor spring forces ahead
			if (p < (dev_Params._NP - 1))
			{
				int pp1 = id + 1;
				float rnab[_D_], dr[_D_];
				float _r, _f;
				for_D_ rnab[d] = r[pp1 + d*rshift];
				_r = sqrt(CalculateRR(rid, rnab, dr));
				_f = -(dev_Params._K1 * (_r - dev_Params._L1)) / _r;
				for_D_ fid[d] -= _f * dr[d];
			}

		//.. 1st neighbor spring forces behind
		if (p > 0)
		{
			int pm1 = id - 1;
			float rnab[_D_], dr[_D_];
			float _r, _f;
			for_D_ rnab[d] = r[pm1 + d*rshift];
			_r = sqrt(CalculateRR(rid, rnab, dr));
			_f = -(dev_Params._K1 * (_r - dev_Params._L1)) / _r;
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. LJ between inter-worm particles
		for (int id2 = w*dev_Params._NP; id2 < (w + 1)*dev_Params._NP; id2++)
		{
			int p2 = id2 % dev_Params._NP;
			int sep = abs(p2 - p);
			if (sep <= 2) continue; //.. ignore 1st and 2nd neighbors
			float rnab[3], dr[3];
			float rr, _f;
			for_D_ rnab[d] = r[id2 + d*rshift];
			rr = CalculateRR(rid, rnab, dr);
			if (rr > dev_Params._R2MIN) continue; //.. repulsive only
			_f = CalculateLJ(rr);
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. viscous drag
		for_D_ fid[d] -= dev_Params._GAMMA * vid[d];

		//.. thermal fluctuations
		for_D_ fid[d] += noiseScaler * randNum[id + d*dev_Params._NPARTICLES];

		//.. assign temp fxid and fyid to memory
		for_D_ f[id + d*fshift] += fid[d];
	}
}

void NW::HoegerPhysics::ThermalForces(real * f, int fshift, float * rn, float mag)
{
	NW::HoegerPhysics::__ThermalForces
		<<<launchInfo->ParticleGridStructure, launchInfo->ParticleBlockStructure >>>
		(f, fshift, rn, mag);
}

__global__ void NW::HoegerPhysics::__ThermalForces(real * f, int fshift, float * rn, float mag)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int bid = blockIdx.y;
	if (tid < dev_Params._NPARTICLES) {
		f[tid + bid * fshift] += mag * rn[tid + bid * dev_Params._NPARTICLES];
	}
}

void NW::HoegerPhysics::ParticleParticleForces(real * f, int fshift, real * r, int rshift, int * nlist, int nshift)
{
	NW::HoegerPhysics::__ParticleParticleForces
		<<<launchInfo->ParticleGridStructure, launchInfo->ParticleBlockStructure >>>
		(r, fshift, r, rshift, nlist, nshift);
}

__global__ void NW::HoegerPhysics::__ParticleParticleForces(real * f, int fshift, real * r, int rshift, int * nlist, int nshift)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES) {

		float fid[_D_], rid[_D_], dr[_D_], _r[_D_];
		float _f, _rr;
		const int np = dev_Params._NP;
		const float drive = dev_Params._DRIVE;
		int pnp = id % np; // particle in chain
		for_D_{ fid[d] = 0.0f; rid[d] = r[id + d*rshift]; }

		bool extensile = dev_Params._EXTENSILE;

		//.. loop through neighbors
		int nnp, nid;
		for (int i = 0; i < dev_Params._NMAX; i++)
		{
			//.. neighbor id
			nid = nlist[id + i*nshift];
			nnp = nid % np;

			//.. if no more players
			if (nid == -1) break;

			for_D_ _r[d] = r[nid + d*rshift];
			_rr = CalculateRR(rid, _r, dr);

			//.. stop if too far
			if (_rr > dev_Params._R2CUT) continue;

			//.. calculate LJ force
			_f = CalculateLJ(_rr);

			//.. extensile driving mechanism applied here!!! need a better place
			if (extensile) {
				if ((pnp < np - 1) && (nnp < np - 1)) {
					float rnxt1[_D_], rnxt2[_D_], x[_D_], u[_D_], f_ext[_D_];
					for_D_ rnxt1[d] = r[(id + 1) + d*rshift];
					for_D_ rnxt2[d] = r[(nid + 1) + d*rshift];
					CalculateRR(rid, rnxt1, x); // vector connecting id to next particle
					CalculateRR(_r, rnxt2, u); // vector connecting nid to next particle

											   //..  average of x and -u (to preserve detailed balance)
					for_D_ f_ext[d] = (x[d] - u[d]) / 2.0f;
					for_D_ fid[d] += (f_ext[d] * dev_Params._DRIVE) / sqrt(_rr); // try this for now
																				 //float dotprod = dot(x, u); // dot product
																				 //if (dotprod < -75.0f){ // if anti-parallel
																				 //	for_D_ fid[d] += dotprod * drive * 
																				 //}
				}
			}

			//.. apply forces to directions
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. assign tmp to memory
		for_D_ f[id + d*fshift] += fid[d];

#ifdef __PRINT_FORCES__
		if (id == __PRINT_INDEX__)
			printf("\n\tLJ Kernel:\n\tf = { %f, %f, %f }", f[id], f[id + fshift], f[id + 2 * fshift]);
#endif
	}
}

void NW::HoegerPhysics::ActiveForces(real * f, int fshift, real * r, int rshift, bool * alive, int * dir)
{
	NW::HoegerPhysics::__ActiveForces
		<< <launchInfo->ParticlesGridStructure, launchInfo->ParticlesBlockStructure >> >
		(f, fshift, r, rshift, alive, dir);
}

__global__ void NW::HoegerPhysics::__ActiveForces(real * f, int fshift, real * r, int rshift, bool * alive, int * dir)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (!dev_Params._EXTENSILE) {
		if (id < dev_Params._NPARTICLES) {
			const int w = id / dev_Params._NP; // worm id
			const int p = id % dev_Params._NP; // particle in chain
			if ((alive[w]) && (p < dev_Params._NP - 2)) { // if not head
				float u[_D_], dr[_D_], rid[_D_], rnab[_D_], umag;
				for_D_ rid[d] = r[id + d*rshift]; // get pos of id particle
				for_D_ rnab[d] = r[(id + 2) + d*rshift]; // get pos of next in chain
				umag = sqrt(CalculateRR(rid, rnab, dr)); // calculate displacement vector and mag
				for_D_ u[d] = dir[w] * (dr[d] / umag); // make unit vector with direction
													   //if (_D_ == 2) Rotate2D(u, dev_Params._DRIVE_ROT); // works for 2d only
				for_D_ f[(id + 1) + d*fshift] += dev_Params._DRIVE * u[d]; // apply drive along unit vector
			}
		}
	}
}

void NW::HoegerPhysics::ExternalForces(real * f, int fshift, real * r, int rshift)
{
	NW::HoegerPhysics::__ExternalForces
		<< <launchInfo->ParticleGridStructure, launchInfo->ParticleBlockStructure >> >
		(f, fshift, r, rshift);
}

__global__ void NW::HoegerPhysics::__ExternalForces(real * f, int fshift, real * r, int rshift)
{
	return __storage__ void();
}

void NW::HoegerPhysics::ContainerForces(real * f, int fshift, real * r, int rshift)
{
	NW::HoegerPhysics::__ContainerForces
		<< <launchInfo->ParticleGridStructure, launchInfo->ParticleBlockStructure >> >
		(f, fshift, r, rshift);
}

__global__ void NW::HoegerPhysics::__ContainerForces(real * f, int fshift, real * r, int rshift)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES) {

		////.. position of giant attractor
		//const float eps = 1.0f;
		//const float sig = 20.0f;
		//const float X = dev_simParams._XBOX / 2.0f;
		//const float Y = dev_simParams._YBOX / 2.0f;
		//const float Z = 0.0f;

		//.. position vectors
		float rid[_D_], rnab[_D_], dr[_D_];
		for_D_{
			rid[d] = r[id + d*rshift];
		rnab[d] = dev_simParams._BOX[d] / 2.0f; // attract to center		
		}

		CalculateRR(rid, rnab, dr); // calculate distance
		BC_dr(dr, dev_simParams._BOX); // boundary conditions
		for_D_ f[id + d*fshift] += dev_Params._LANDSCALE * dr[d]; // calc and apply force

		//.. harmonic potential zeroed around z = 0
		//f[id + 2 * fshift] -= dev_Params._LANDSCALE * r[id + 2 * rshift];
		//
		//.. attraction to giant attractor at { Lx/2, Ly/2, 0 }
		//const float rr = CalculateRR_3d(rid, rnab, dr);
		//const float _f = CalculateLJ_3d(rr, sig, eps);
		//for (int d = 0; d < 3; d++) 
		//	f[id + d * fshift] -= _f * dr[d];
	}
}

void NW::HoegerPhysics::BondBendingForces(real * f, int fshift, real * r, int rshift)
{
	NW::HoegerPhysics::__BondBendingForces
		<< <launchInfo->BodiesGridStructure, launchInfo->BodiesBlockStructure >> >
		(f, fshift, r, rshift);
}

__global__ void NW::HoegerPhysics::__BondBendingForces(real * f, int fshift, real * r, int rshift)
{
	int wid = threadIdx.x + blockDim.x * blockIdx.x;
	int nworms = dev_Params._NWORMS;
	int np = dev_Params._NP;
	const float ka_ratio = dev_Params._Ka_RATIO;
	const float ka1 = dev_Params._Ka;
	const float ka2 = dev_Params._Ka2;
	if (wid < nworms) {

		//.. chose ka to use (default to ka1)
		const float k_a = (wid <= nworms*ka_ratio ? ka1 : ka2);

		//.. loop through particles in worm (excluding last)
		int p2id, p3id;
		float r1[_D_], r2[_D_], r3[_D_], f1[_D_], f2[_D_], f3[_D_];
		float r12[_D_], r23[_D_];
		//float BOX[] = { dev_simParams._XBOX, dev_simParams._YBOX };
		for (int p1id = wid * np; p1id < ((wid + 1)*np) - 2; p1id++) {

			//.. particle ids
			p2id = p1id + 1;
			p3id = p2id + 1;

			//.. grab memory and calculate distances
			for_D_{

				//.. memory
				r1[d] = r[p1id + d *rshift];
			r2[d] = r[p2id + d *rshift];
			r3[d] = r[p3id + d *rshift];

			//.. distances
			r12[d] = r2[d] - r1[d];
			r23[d] = r3[d] - r2[d];
			}

				//.. boundary conditions
			BC_dr(r12, dev_simParams._BOX);
			BC_dr(r23, dev_simParams._BOX);

			//.. calculate terms
			float dot_r12_r23 = dot(r12, r23);
			float r12r12 = dot(r12, r12);
			float r23r23 = dot(r23, r23);
			float mag12inv = 1.0f / mag(r12);
			float mag23inv = 1.0f / mag(r23);
			float a = k_a * mag12inv * mag23inv;
			float A[] = { a, a, a }; // always 3d

									 //..  calculate forces x, y, z
			for_D_{

				//.. particle 1
				f1[d] = A[d] * (r23[d] - ((dot_r12_r23 / r12r12) * r12[d]));

			//.. particle 2
			f2[d] = A[d] * (((dot_r12_r23 / r12r12) * r12[d]) - ((dot_r12_r23 / r23r23) * r23[d]) + r12[d] - r23[d]);

			//.. particle 3
			f3[d] = A[d] * (((dot_r12_r23 / r23r23) * r23[d]) - r12[d]);

			//.. apply forces to all 3 particles
			f[p1id + d * fshift] -= f1[d];
			f[p2id + d * fshift] -= f2[d];
			f[p3id + d * fshift] -= f3[d];
			}
		}
	}
}

void NW::HoegerPhysics::UpdateSystem(real * f, int fshift, real * f_old, int foshift, real * v, int vshift, real * r, int rshift, int * cell, int cshift, real dt)
{
	NW::HoegerPhysics::__UpdateSystem
		<< < launchInfo->ParticleGridStructure, launchInfo->ParticleBlockStructure >> >
		(f, fshift, f_old, foshift, v, vshift, r, rshift, cell, cshift, dt);
}

__global__ void NW::HoegerPhysics::__UpdateSystem(real * f, int fshift, real * f_old, int foshift, real * v, int vshift, real * r, int rshift, int * cell, int cshift, real dt)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < dev_Params._NPARTICLES)
	{
		//.. local components
		float dv[_D_], dr[_D_], rid[_D_], fid[_D_];
		for_D_ rid[d] = r[id + d*rshift];
		for_D_ fid[d] = f[id + d*fshift];

		//.. boundary conditions
		BC_r(fid, rid, dev_simParams._BOX);

		//.. change in velocity
		for_D_ dv[d] = 0.5f * (fid[d] + f_old[id + d*foshift]) * dt;

		//.. change in position
		for_D_ dr[d] = v[id + d*vshift] * dt + 0.5f * f_old[id + d*foshift] * dt * dt;

		//.. save forces
		for_D_ f_old[id + d*foshift] = fid[d];

		//.. update positions
		for_D_ rid[d] += dr[d];

		//.. boundary conditions and apply new pos
		//BC_r(rid, dev_simParams._BOX);
		for_D_ r[id + d*rshift] = rid[d];

		//.. update velocities
		for_D_ v[id + d*vshift] += dv[d];

		//.. update cell address
		for_D_ cell[id + d*cshift] = (int)(rid[d] / dev_Params._DCELL);
	}
}

void NW::HoegerPhysics::UpdateSystemFast(real * f, int fshift, real * f_old, int foshift, real * v, int vshift, real * r, int rshift, int * cell, int cshift, real dt)
{
	NW::HoegerPhysics::__UpdateSystemFast
		<<<launchInfo->ParticleGridStructure, launchInfo->ParticleBlockStructure >>>
		(f, fshift, f_old, foshift, v, vshift, r, rshift, cell, cshift, dt);
}

__global__ void NW::HoegerPhysics::__UpdateSystemFast(real * f, int fshift, real * f_old, int foshift, real * v, int vshift, real * r, int rshift, int * cell, int cshift, real dt)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int bid = blockIdx.y;
	if (tid < dev_Params._NPARTICLES) {

		const int fid = tid + bid * fshift;
		const int foid = tid + bid * foshift;
		const int vid = tid + bid * vshift;
		const int rid = tid + bid * rshift;
		const int cid = tid + bid * cshift;

		//.. boundary conditions
		BC_r(f[fid], r[rid], dev_simParams._BOX[bid], bid); // only applies to

		float dvx = 0.5f * (f[fid] + f_old[foid]) * dt;
		float dx = v[vid] * dt + 0.5f * f_old[foid] * dt * dt;
		f_old[foid] = f[fid];
		r[rid] += dx;
		v[vid] += dvx;

		//.. boundary conditions
		//BC_r(r[rid], dev_simParams._BOX[bid]); // only applies to

		//if (bid == 0) BC_r(r[rid], dev_simParams._XBOX);
		//else if (bid == 1) BC_r(r[rid], dev_simParams._YBOX);
		//else if (bid == 2) BC_r(r[rid], dev_simParams._ZBOX);

		//.. update cell list
		//cell[cid] = (int)(r[rid] / dev_Params._DCELL);

		f[fid] = 0.0f;
	}
}