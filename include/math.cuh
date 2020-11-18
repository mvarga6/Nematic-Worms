#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "dtypes.h"

__host__ __device__
float3 operator*(const float &lhs, const float3 &rhs);
__host__ __device__
float3 operator*(const float3 &lhs, const float3 &rhs);
__host__ __device__
float3 operator-(const float3 &lhs, const float3 &rhs);
__host__ __device__
void operator+=(float3 &lhs, const float3 &rhs);
__host__ __device__
void operator-=(float3 &lhs, const float3 &rhs);
__host__ __device__
float dot(float3 a, float3 b);
__host__ __device__
float distance(float3 pt1, float3 pt2, float3 *dr = NULL);
__host__ __device__ void displacement_pbc(float3 p1, float3 p2, float3 box, float* distance, float3* displacement);
__host__ __device__ void displacement_pbc_x(float3 p1, float3 p2, float3 box, float* distance, float3* displacement);
__host__ __device__ void displacement_pbc_y(float3 p1, float3 p2, float3 box, float* distance, float3* displacement);
__host__ __device__ void displacement_pbc_z(float3 p1, float3 p2, float3 box, float* distance, float3* displacement);
__host__ __device__ void apply_pbc(float3 box, float3* p);
__host__ __device__ void apply_pbc_x(float3 box, float3* p);
__host__ __device__ void apply_pbc_y(float3 box, float3* p);
__host__ __device__ void apply_pbc_z(float3 box, float3* p);


// __device__ PositionFunction p_apply_pbc;
// __device__ PositionFunction p_apply_pbc_x;
// __device__ PositionFunction p_apply_pbc_y;
// __device__ PositionFunction p_apply_pbc_z;

// __device__ DistanceFunction p_displacement_pbc;
// __device__ DistanceFunction p_displacement_pbc_x;
// __device__ DistanceFunction p_displacement_pbc_y;
// __device__ DistanceFunction p_displacement_pbc_z;

extern PositionFunction h_apply_pbc;
extern PositionFunction h_apply_pbc_x;
extern PositionFunction h_apply_pbc_y;
extern PositionFunction h_apply_pbc_z;

__host__ void copy_device_function_symbols();