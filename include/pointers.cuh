#pragma once
#include <cuda.h>
#include "dtypes.h"

__device__ PositionFunction p_apply_pbc   = apply_pbc;
__device__ PositionFunction p_apply_pbc_x = apply_pbc_x;
__device__ PositionFunction p_apply_pbc_y = apply_pbc_y;
__device__ PositionFunction p_apply_pbc_z = apply_pbc_z;

__device__ DistanceFunction p_displacement_pbc   = displacement_pbc;
__device__ DistanceFunction p_displacement_pbc_x = displacement_pbc_x;
__device__ DistanceFunction p_displacement_pbc_y = displacement_pbc_y;
__device__ DistanceFunction p_displacement_pbc_z = displacement_pbc_z;
