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


extern PositionFunction h_position_function_table[];
__constant__ extern PositionFunction d_position_function_table[4];
extern DistanceFunction h_distance_function_table[];
__constant__ extern DistanceFunction d_distance_function_table[4];

__host__ void SetupDeviceFunctionTables();
__host__ PositionFunction GetPositionFunction(int i);
__host__ DistanceFunction GetDistanceFunction(int i);