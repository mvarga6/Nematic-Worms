#pragma once

#include "particles.h"
#include "math.cuh"

void FilamentBackboneForce(
    Particles *particles,
    int n_filaments,
    int filament_size,
    float spring_constant,
    float bond_length,
    int threads_per_block
);

void ApplyPositionFunction(
    PositionFunction h_func,
    Particles* particles,
    float3 box,
    int threads_per_block
);