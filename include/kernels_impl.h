#pragma once

#include <cuda.h>
#include "particles.h"

__global__
void FilamentBackboneForceKernel(
    ptype *position,
    ptype *force,
    int n_filaments,
    int filament_size,
    float spring_constant,
    float bond_length
);

