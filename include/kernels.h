#pragma once

#include "particles.h"


void FilamentBackboneForce(
    Particles *particles,
    int n_filaments,
    int filament_size,
    float spring_constant,
    float bond_length,
    int threads_per_block
);
