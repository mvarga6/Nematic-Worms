#pragma once

#include <cuda.h>
#include "particles.h"
#include "math.cuh"


__global__
void FilamentBackboneForceKernel(
    ptype* position,
    ptype* force,
    int n_filaments,
    int filament_size,
    float spring_constant,
    float bond_length
);

__global__
void ApplyPositionFunctionKernel(
    PositionFunction func,
    ptype *position,
    int n_particles,
    float3 box
);