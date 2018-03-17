#pragma once

#include "../defs.h"

namespace NW
{
	//
	// Fills fid with bonding forces from neighbors.
	// EXECUTES: In 'per-particle' force kernel
	//
	typedef void(*FilamentBondingInteraction)(real fid[_D_], const real *r, const int N, const int np, const int id);

	//
	// Fills fworm with forces on all particles
	// in a worm from other particles in the same worm.
	// EXECUTES: In 'per-filament' force kernel
	//
	typedef float(*InterFilamentInteraction)(real *fworm, const real *r, const int N, const int np, const int wid);

	//
	// Given the square of the displacement of a target
	// particle from another in a different filament, calculate
	// the force on the target particle.
	//
	typedef float(*ParticleParticleInteraction)(real RR_int);

	//
	// Given
	//
	typedef float(*ParticleSolventInteraction)(real v[_D_]);


}