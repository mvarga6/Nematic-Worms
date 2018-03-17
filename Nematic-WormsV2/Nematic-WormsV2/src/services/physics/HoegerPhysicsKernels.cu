
#include "../../../include/physics/HoegerPhysicsKernels.cuh"
#include "../../../include/Math.cuh"

__global__ void NW::HoegerInternalForces(
	real *f,
	real *v,
	real *r,
	int N,
	int np,
	real kbond,
	real lbond,
	DistanceSquaredMetric RR,
	InteractionFunction interaction)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < N)
	{

		//.. particel number in worm
		int p = id % np;
		int w = id / np;

		//.. local memory
		float rid[_D_], fid[_D_], vid[_D_];

		//.. init local as needed
		for_D_
		{
			rid[d] = r[id + d * N];
			fid[d] = 0.0f;
			vid[d] = v[id + d * N];
		}

		//.. 1st neighbor spring forces ahead
		if (p < (np - 1))
		{
			int pp1 = id + 1;
			float rnab[_D_], dr[_D_];
			float _r, _f;
			for_D_ rnab[d] = r[pp1 + d * N];
			_r = sqrt(RR(rid, rnab, dr));
			_f = -(kbond * (_r - lbond)) / _r;
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. 1st neighbor spring forces behind
		if (p > 0)
		{
			int pm1 = id - 1;
			float rnab[_D_], dr[_D_];
			float _r, _f;
			for_D_ rnab[d] = r[pm1 + d * N];
			_r = sqrt(RR(rid, rnab, dr));
			_f = -(kbond * (_r - lbond)) / _r;
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. LJ between inter-worm particles
		for (int id2 = w * np; id2 < (w + 1) * np; id2++)
		{
			int p2 = id2 % np;
			int sep = abs(p2 - p);
			if (sep <= 2) continue; //.. ignore 1st and 2nd neighbors
			float rnab[3], dr[3];
			float rr, _f;
			for_D_ rnab[d] = r[id2 + d * N];
			rr = RR(rid, rnab, dr);
			//if (rr > Model.R2Min) continue; //.. repulsive only
			_f = interaction(rr);
			for_D_ fid[d] -= _f * dr[d];
		}

		// TODO: move viscous drag 
		//.. viscous drag (move to interaction with environment)
		//for_D_ fid[d] -= gamma * vid[d];

		//.. thermal fluctuations
		//for_D_ fid[d] += noiseScaler * randNum[id + d * N];

		//.. assign temp fxid and fyid to memory
		for_D_ f[id + d * N] += fid[d];
	}
}

__global__ void NW::HoegerThermalForces(
	real *f,
	int N,
	float * gpu_rn,
	float mag)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int bid = blockIdx.y;
	if (tid < N)
	{
		f[tid + bid * N] += mag * gpu_rn[tid + bid * N];
	}
}

__global__ void NW::HoegerParticleParticleForces(
	real *f,
	real *r,
	int N,
	int np,
	int *nlist,
	int nmax,
	DistanceSquaredMetric RR,
	InteractionFunction interaction)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < N)
	{

		float fid[_D_], rid[_D_], dr[_D_], _r[_D_];
		float _f, _rr;
		const int np = np;
		int pnp = id % np; // particle in chain
		for_D_{ fid[d] = 0.0f; rid[d] = r[id + d*N]; }

			//.. loop through neighbors
		int nnp, nid;
		for (int i = 0; i < nmax; i++)
		{
			//.. neighbor id
			nid = nlist[id + i * N];
			nnp = nid % np;

			//.. if no more players
			if (nid == -1) break;

			for_D_ _r[d] = r[nid + d*N];
			_rr = RR(rid, _r, dr);

			//.. stop if too far
			//if (_rr > Model.R2Cutoff) continue;

			//.. calculate LJ force
			_f = interaction(_rr);

			//.. apply forces to directions
			for_D_ fid[d] -= _f * dr[d];
		}

		//.. assign tmp to memory
		for_D_ f[id + d*N] += fid[d];

#ifdef __PRINT_FORCES__
		if (id == __PRINT_INDEX__)
			printf("\n\tLJ Kernel:\n\tf = { %f, %f, %f }", f[id], f[id + fshift], f[id + 2 * fshift]);
#endif
	}
}

__global__ void NW::HoegerActiveForces(
	real *f,
	real *r,
	int N,
	int np,
	real * drive,
	DistanceSquaredMetric RR)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id < N)
	{
		const int w = id / np; // worm id
		const int p = id % np; // particle in chain
		if (p > 0 && (p < np - 1)) // if not head or tail
		{
			float u[_D_], dr[_D_], rid[_D_], rnab[_D_], umag;
			for_D_ rid[d] = r[(id - 1) + d * N]; // get pos of particle behind
			for_D_ rnab[d] = r[(id + 1) + d * N]; // get pos of next in chain
			umag = sqrt(RR(rid, rnab, dr)); // calculate displacement vector and mag
			for_D_ u[d] = (dr[d] / umag); // make unit vector with direction
			for_D_ f[id + d * N] += drive[w] * u[d]; // apply drive along unit vector
		}
	}
}

__global__ void NW::HoegerExternalForces(
	real *f,
	real *v,
	int N,
	real gamma)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < N)
	{
		for_D_ f[id + d*N] -= gamma * v[id + d*N];
	}
}

__global__ void NW::HoegerContainerForces(
	real *f,
	real *r,
	int N,
	BoundaryForce container_interaction)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < N)
	{
		float rid[_D_], f_cont[_D_];

		for_D_
		{
			rid[d] = r[id + d*N];
		f_cont[d] = 0;
		}

		container_interaction(rid, f_cont);
		for_D_ f[id + d*N] += f_cont[d];
	}
}

__global__ void NW::HoegerBondBendingForces(
	real *f,
	real *r,
	int N,
	int nworms,
	int np,
	real kbend,
	DisplacementMetric disp_metric)
{
	const int wid = threadIdx.x + blockDim.x * blockIdx.x;

	if (wid < nworms)
	{
		const int shift = N;

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
			for_D_
			{
				//.. memory
				r1[d] = r[p1id + d * shift];
			r2[d] = r[p2id + d * shift];
			r3[d] = r[p3id + d * shift];

			//.. distances
			r12[d] = r2[d] - r1[d];
			r23[d] = r3[d] - r2[d];
			}

				//.. boundary conditions
			disp_metric(r12);
			disp_metric(r23);
			//BC_dr(r12, dev_simParams._BOX);
			//BC_dr(r23, dev_simParams._BOX);

			//.. calculate terms
			float dot_r12_r23 = NW::Math::DotProd(r12, r23);
			float r12r12 = NW::Math::DotProd(r12, r12);
			float r23r23 = NW::Math::DotProd(r23, r23);
			float mag12inv = 1.0f / NW::Math::Mag(r12);
			float mag23inv = 1.0f / NW::Math::Mag(r23);
			float a = kbend * mag12inv * mag23inv;
			float A[] = { a, a, a }; // always 3d

									 //..  calculate forces x, y, z
			for_D_
			{
				//.. particle 1
				f1[d] = A[d] * (r23[d] - ((dot_r12_r23 / r12r12) * r12[d]));

			//.. particle 2
			f2[d] = A[d] * (((dot_r12_r23 / r12r12) * r12[d]) - ((dot_r12_r23 / r23r23) * r23[d]) + r12[d] - r23[d]);

			//.. particle 3
			f3[d] = A[d] * (((dot_r12_r23 / r23r23) * r23[d]) - r12[d]);

			//.. apply forces to all 3 particles
			f[p1id + d * shift] -= f1[d];
			f[p2id + d * shift] -= f2[d];
			f[p3id + d * shift] -= f3[d];
			}
		}
	}
}

__global__ void NW::HoegerUpdateSystem(
	real *f,
	real *f_old,
	real *v,
	real *r,
	int N,
	BoundaryFunction boundary_condition,
	real dt)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < N)
	{
		//.. local components
		float dv[_D_], dr[_D_], rid[_D_], fid[_D_];
		for_D_ rid[d] = r[id + d * N];
		for_D_ fid[d] = f[id + d * N];

		//.. boundary conditions
		//BC_r(fid, rid, EnvBox.L);

		//.. change in velocity
		for_D_ dv[d] = 0.5f * (fid[d] + f_old[id + d * N]) * dt;

		//.. change in position
		for_D_ dr[d] = v[id + d * N] * dt + 0.5f * f_old[id + d * N] * dt * dt;

		//.. save forces
		for_D_ f_old[id + d * N] = fid[d];

		//.. update positions
		for_D_ rid[d] += dr[d];

		// handle boundary conditions
		boundary_condition(rid);

		//.. boundary conditions and apply new pos
		//BC_r(rid, dev_simParams._BOX);
		for_D_ r[id + d * N] = rid[d];

		//.. update velocities
		for_D_ v[id + d * N] += dv[d];

		//.. update cell address
		//for_D_ cell[id + d*cshift] = (int)(rid[d] / dev_Params._DCELL);
	}
}

//__global__ void NW::HoegerPhysicsModel::__UpdateSystemFast(real *f, real *f_old, real* v, real * r, int N, real dt)
//{
//	int tid = threadIdx.x + blockDim.x * blockIdx.x;
//	int bid = blockIdx.y;
//	if (tid < N) {
//
//		const int fid = tid + bid * N;
//		const int foid = tid + bid * N;
//		const int vid = tid + bid * N;
//		const int rid = tid + bid * N;
//		const int cid = tid + bid * cshift;
//
//		//.. boundary conditions
//		BC_r(f[fid], r[rid], EnvBox.L[bid], bid); // only applies to
//
//		float dvx = 0.5f * (f[fid] + f_old[foid]) * dt;
//		float dx = v[vid] * dt + 0.5f * f_old[foid] * dt * dt;
//		f_old[foid] = f[fid];
//		r[rid] += dx;
//		v[vid] += dvx;
//
//		//.. boundary conditions
//		//BC_r(r[rid], dev_simParams._BOX[bid]); // only applies to
//
//		//if (bid == 0) BC_r(r[rid], dev_simParams._XBOX);
//		//else if (bid == 1) BC_r(r[rid], dev_simParams._YBOX);
//		//else if (bid == 2) BC_r(r[rid], dev_simParams._ZBOX);
//
//		//.. update cell list
//		//cell[cid] = (int)(r[rid] / dev_Params._DCELL);
//
//		f[fid] = 0.0f;
//	}
//}